//! Byte-fallback, score-driven BPE for the legacy `llama2.c` tokenizer format.
//!
//! The binary contains a little-endian maximum piece length followed by
//! `(score: f32, byte_length: i32, UTF-8 piece)` for every vocabulary entry.
//! Byte fallback tokens occupy IDs 3 through 258, as in the reference runtime.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

const BOS: usize = 1;
const EOS: usize = 2;
const BYTE_OFFSET: usize = 3;
const MAX_VOCABULARY: usize = 1_000_000;
const MAX_PIECE_BYTES: usize = 1_048_576;
const MAX_TOKENIZER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_INPUT_BYTES: usize = 1_048_576;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pieces: Vec<String>,
    scores: Vec<f32>,
    lookup: HashMap<String, usize>,
}

impl Tokenizer {
    pub fn from_file(path: impl AsRef<Path>, vocab_size: usize) -> Result<Self, String> {
        let path = path.as_ref();
        let metadata = fs::metadata(path)
            .map_err(|error| format!("Cannot read tokenizer {}: {error}", path.display()))?;
        if metadata.len() > MAX_TOKENIZER_BYTES {
            return Err("Tokenizer file exceeds the 256 MiB safety limit".into());
        }
        let bytes = fs::read(path)
            .map_err(|error| format!("Cannot read tokenizer {}: {error}", path.display()))?;
        Self::from_bytes(&bytes, vocab_size)
    }

    pub fn from_bytes(bytes: &[u8], vocab_size: usize) -> Result<Self, String> {
        if !(3..=MAX_VOCABULARY).contains(&vocab_size) {
            return Err("Tokenizer vocabulary size must be between 3 and 1,000,000".into());
        }
        if bytes.len() as u64 > MAX_TOKENIZER_BYTES {
            return Err("Tokenizer file exceeds the 256 MiB safety limit".into());
        }
        let mut cursor = Cursor { bytes, offset: 0 };
        let max_piece_length = cursor.i32()?;
        if max_piece_length <= 0 || max_piece_length as usize > MAX_PIECE_BYTES {
            return Err("Invalid tokenizer maximum piece length".into());
        }
        // Even empty pieces need eight header bytes. Check before allocating.
        if vocab_size > (bytes.len().saturating_sub(4)) / 8 {
            return Err("Tokenizer is truncated for the requested vocabulary size".into());
        }
        let mut pieces = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        let mut lookup = HashMap::with_capacity(vocab_size);
        for index in 0..vocab_size {
            let score = f32::from_le_bytes(cursor.take(4)?.try_into().unwrap());
            if !score.is_finite() {
                return Err(format!("Tokenizer score {index} is not finite"));
            }
            let length = cursor.i32()?;
            if length < 0 || length as usize > max_piece_length as usize {
                return Err(format!("Invalid tokenizer piece length at token {index}"));
            }
            let raw = cursor.take(length as usize)?;
            let piece = std::str::from_utf8(raw)
                .map_err(|_| format!("Tokenizer piece {index} is not valid UTF-8"))?
                .to_owned();
            // Valid SentencePiece vocabularies normally have unique pieces.
            // Keeping the first ID makes duplicate decoded spellings stable.
            lookup.entry(piece.clone()).or_insert(index);
            pieces.push(piece);
            scores.push(score);
        }
        if cursor.offset != bytes.len() {
            return Err("Tokenizer has trailing bytes or an incorrect vocabulary size".into());
        }
        Ok(Self {
            pieces,
            scores,
            lookup,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Encode with the same dummy-space prefix and merge ordering as `llama2.c`.
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Result<Vec<usize>, String> {
        if text.len() > MAX_INPUT_BYTES {
            return Err("Tokenizer input exceeds the 1 MiB safety limit".into());
        }
        let mut tokens = Vec::with_capacity(text.len().saturating_add(2));
        if bos {
            tokens.push(BOS);
        }
        if !text.is_empty() {
            let space = self.lookup.get(" ").copied().ok_or_else(|| {
                "Tokenizer vocabulary is missing the dummy-prefix space token".to_owned()
            })?;
            tokens.push(space);
        }
        for character in text.chars() {
            let mut raw = [0_u8; 4];
            let piece = character.encode_utf8(&mut raw);
            if let Some(&id) = self.lookup.get(piece) {
                tokens.push(id);
            } else {
                for byte in piece.as_bytes() {
                    let id = *byte as usize + BYTE_OFFSET;
                    if id >= self.pieces.len() {
                        return Err(format!(
                            "Tokenizer lacks byte-fallback token for byte 0x{byte:02X}"
                        ));
                    }
                    if parse_byte_piece(&self.pieces[id]) != Some(*byte) {
                        return Err(format!(
                            "Tokenizer has an incompatible byte-fallback token at ID {id}"
                        ));
                    }
                    tokens.push(id);
                }
            }
        }
        loop {
            let mut best: Option<(usize, usize, f32)> = None;
            for index in 0..tokens.len().saturating_sub(1) {
                let left = &self.pieces[tokens[index]];
                let right = &self.pieces[tokens[index + 1]];
                let mut pair = String::with_capacity(left.len() + right.len());
                pair.push_str(left);
                pair.push_str(right);
                if let Some(&id) = self.lookup.get(&pair) {
                    let score = self.scores[id];
                    // Strict comparison preserves the first pair on score ties.
                    if best.is_none_or(|(_, _, previous_score)| score > previous_score) {
                        best = Some((index, id, score));
                    }
                }
            }
            let Some((index, id, _)) = best else {
                break;
            };
            tokens[index] = id;
            tokens.remove(index + 1);
        }
        if eos {
            tokens.push(EOS);
        }
        Ok(tokens)
    }

    /// Return raw bytes so byte-fallback pieces can be joined before decoding.
    pub fn decode_piece(&self, previous_token: usize, token: usize) -> Result<Vec<u8>, String> {
        let piece = self
            .pieces
            .get(token)
            .ok_or_else(|| format!("Token ID {token} is outside the vocabulary"))?;
        let piece = if previous_token == BOS {
            piece.strip_prefix(' ').unwrap_or(piece)
        } else {
            piece
        };
        if let Some(byte) = parse_byte_piece(piece) {
            Ok(vec![byte])
        } else {
            Ok(piece.as_bytes().to_vec())
        }
    }

    /// Human-readable vocabulary spelling, including explicit byte pieces.
    pub fn token_text(&self, token: usize) -> String {
        match token {
            BOS => "<BOS>".into(),
            EOS => "<EOS>".into(),
            _ => self
                .pieces
                .get(token)
                .cloned()
                .unwrap_or_else(|| format!("<invalid:{token}>")),
        }
    }

    pub fn decode_bytes(&self, tokens: &[usize]) -> Result<Vec<u8>, String> {
        let mut bytes = Vec::new();
        let mut previous = usize::MAX;
        for &token in tokens {
            if token != BOS && token != EOS {
                bytes.extend_from_slice(&self.decode_piece(previous, token)?);
            } else if token >= self.pieces.len() {
                return Err(format!("Token ID {token} is outside the vocabulary"));
            }
            previous = token;
        }
        Ok(bytes)
    }

    /// Complete UTF-8 byte pieces round-trip exactly. An unfinished generation
    /// may end midway through a character; replacement text is used only then.
    pub fn decode(&self, tokens: &[usize]) -> Result<String, String> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(tokens)?).into_owned())
    }
}

fn parse_byte_piece(piece: &str) -> Option<u8> {
    if piece.len() != 6 || !piece.starts_with("<0x") || !piece.ends_with('>') {
        return None;
    }
    u8::from_str_radix(&piece[3..5], 16).ok()
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8], String> {
        let end = self
            .offset
            .checked_add(length)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| "Tokenizer file is truncated".to_owned())?;
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocabulary(extra: &[(&str, f32)]) -> Vec<(String, f32)> {
        let mut pieces = vec![
            ("<unk>".into(), 0.0),
            ("\n<s>\n".into(), 0.0),
            ("\n</s>\n".into(), 0.0),
        ];
        pieces.extend((0_u16..=255).map(|byte| (format!("<0x{byte:02X}>"), -100.0)));
        pieces.push((" ".into(), -10.0));
        pieces.extend(
            extra
                .iter()
                .map(|&(piece, score)| (piece.to_owned(), score)),
        );
        pieces
    }

    fn binary(pieces: &[(String, f32)]) -> Vec<u8> {
        let max = pieces.iter().map(|(piece, _)| piece.len()).max().unwrap();
        let mut bytes = (max as i32).to_le_bytes().to_vec();
        for (piece, score) in pieces {
            bytes.extend_from_slice(&score.to_le_bytes());
            bytes.extend_from_slice(&(piece.len() as i32).to_le_bytes());
            bytes.extend_from_slice(piece.as_bytes());
        }
        bytes
    }

    fn tokenizer(extra: &[(&str, f32)]) -> Tokenizer {
        let pieces = vocabulary(extra);
        Tokenizer::from_bytes(&binary(&pieces), pieces.len()).unwrap()
    }

    #[test]
    fn byte_fallback_joins_utf8_before_decoding() {
        let tokenizer = tokenizer(&[]);
        let text = "café 🦀 日本語\n";
        let ids = tokenizer.encode(text, true, true).unwrap();
        assert_eq!(ids.first(), Some(&BOS));
        assert_eq!(ids.last(), Some(&EOS));
        assert!(ids.contains(&(0xF0 + BYTE_OFFSET)));
        assert_eq!(tokenizer.decode(&ids).unwrap(), text);
        assert_eq!(tokenizer.decode_bytes(&ids).unwrap(), text.as_bytes());
    }

    #[test]
    fn uses_highest_scoring_pair_before_earlier_lower_score_pair() {
        let tokenizer = tokenizer(&[
            ("a", -5.0),
            ("b", -5.0),
            ("c", -5.0),
            ("ab", 1.0),
            ("bc", 2.0),
        ]);
        let ids = tokenizer.encode("abc", true, false).unwrap();
        let pieces: Vec<_> = ids.iter().map(|&id| tokenizer.token_text(id)).collect();
        assert_eq!(pieces, ["<BOS>", " ", "a", "bc"]);
        assert_eq!(tokenizer.decode(&ids).unwrap(), "abc");
    }

    #[test]
    fn merge_ties_choose_the_leftmost_pair() {
        let tokenizer = tokenizer(&[
            ("a", -5.0),
            ("b", -5.0),
            ("c", -5.0),
            ("ab", 2.0),
            ("bc", 2.0),
        ]);
        let ids = tokenizer.encode("abc", true, false).unwrap();
        assert_eq!(tokenizer.token_text(ids[2]), "ab");
        assert_eq!(tokenizer.token_text(ids[3]), "c");
    }

    #[test]
    fn merges_the_dummy_prefix_and_only_strips_it_after_bos() {
        let tokenizer = tokenizer(&[("a", -1.0), (" a", 2.0)]);
        let ids = tokenizer.encode("a", true, false).unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(tokenizer.decode(&ids).unwrap(), "a");
        assert_eq!(tokenizer.decode_piece(0, ids[1]).unwrap(), b" a");
        assert_eq!(tokenizer.decode_piece(BOS, ids[1]).unwrap(), b"a");
        assert_eq!(tokenizer.encode("", true, true).unwrap(), [BOS, EOS]);
        assert!(tokenizer.encode("", false, false).unwrap().is_empty());
    }

    #[test]
    fn known_unicode_pieces_are_not_split_into_bytes() {
        let tokenizer = tokenizer(&[("é", 1.0)]);
        let ids = tokenizer.encode("é", true, false).unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(tokenizer.token_text(ids[2]), "é");
        assert_eq!(tokenizer.decode(&ids).unwrap(), "é");
    }

    #[test]
    fn invalid_ids_and_incomplete_utf8_do_not_panic() {
        let tokenizer = tokenizer(&[]);
        assert!(tokenizer.decode_piece(0, usize::MAX).is_err());
        assert!(tokenizer.decode(&[usize::MAX]).is_err());
        assert_eq!(
            tokenizer.token_text(usize::MAX),
            format!("<invalid:{}>", usize::MAX)
        );
        let partial = [BOS, 0xF0 + BYTE_OFFSET];
        assert_eq!(tokenizer.decode_bytes(&partial).unwrap(), [0xF0]);
        assert_eq!(tokenizer.decode(&partial).unwrap(), "\u{fffd}");
    }

    #[test]
    fn malformed_binary_is_rejected_before_allocation_or_indexing() {
        let pieces = vocabulary(&[]);
        let good = binary(&pieces);
        assert!(Tokenizer::from_bytes(&[], pieces.len()).is_err());
        assert!(Tokenizer::from_bytes(&good[..good.len() - 1], pieces.len()).is_err());
        assert!(Tokenizer::from_bytes(&good, usize::MAX).is_err());
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(Tokenizer::from_bytes(&trailing, pieces.len()).is_err());
        let mut invalid_score = good.clone();
        invalid_score[4..8].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(Tokenizer::from_bytes(&invalid_score, pieces.len()).is_err());
        let mut invalid_length = good.clone();
        invalid_length[8..12].copy_from_slice(&(-1_i32).to_le_bytes());
        assert!(Tokenizer::from_bytes(&invalid_length, pieces.len()).is_err());
        let mut invalid_utf8 = good;
        invalid_utf8[12] = 0xFF;
        assert!(Tokenizer::from_bytes(&invalid_utf8, pieces.len()).is_err());
    }
}
