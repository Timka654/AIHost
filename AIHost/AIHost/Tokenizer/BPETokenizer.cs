using AIHost.ICompute.Vulkan;

namespace AIHost.Tokenizer;

/// <summary>
/// BPE (Byte Pair Encoding) Tokenizer for GGUF models
/// </summary>
public class BPETokenizer
{
    private readonly string[] _tokens;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly int _bosToken;
    private readonly int _eosToken;
    private readonly int _unknownToken;
    private readonly bool _isSentencePiece;

    private static readonly ILogger _logger = AppLogger.Create<BPETokenizer>();

    // GPT-2 style bytes_to_unicode mapping for BPE/tiktoken tokenizers
    // Maps each byte (0-255) to a unicode character that exists as a token in the vocabulary
    private readonly Dictionary<byte, int> _byteToTokenId;
    // Inverse mapping: GPT-2 bytes_to_unicode char → original byte (for decoding)
    private readonly Dictionary<char, byte> _charToByte;

    public int VocabSize => _tokens.Length;
    public int BosToken => _bosToken;
    public int EosToken => _eosToken;

    public BPETokenizer(string[] tokens, int bosToken = 1, int eosToken = 2, int unknownToken = 0, bool isSentencePiece = true)
    {
        _tokens = tokens;
        _bosToken = bosToken;
        _eosToken = eosToken;
        _unknownToken = unknownToken;
        _isSentencePiece = isSentencePiece;

        // Build token -> id lookup
        _tokenToId = new Dictionary<string, int>();
        for (int i = 0; i < tokens.Length; i++)
        {
            _tokenToId[tokens[i]] = i;
        }

        // Build byte -> token id mapping for BPE/tiktoken (GPT-2 style)
        _byteToTokenId = BuildByteToTokenIdMapping();
        // Build inverse: GPT-2 unicode char -> byte (for decoding token text back to UTF-8)
        _charToByte = BuildCharToByteMapping();
    }

    /// <summary>
    /// Builds the inverse of the GPT-2 bytes_to_unicode mapping: unicode char → original byte.
    /// Used to decode BPE token strings back to raw bytes which are then UTF-8 decoded.
    /// </summary>
    private static Dictionary<char, byte> BuildCharToByteMapping()
    {
        var byteList = new List<int>();
        var charList = new List<int>();
        for (int i = 33; i <= 126; i++) { byteList.Add(i); charList.Add(i); }
        for (int i = 161; i <= 172; i++) { byteList.Add(i); charList.Add(i); }
        for (int i = 174; i <= 255; i++) { byteList.Add(i); charList.Add(i); }
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!byteList.Contains(b))
            { byteList.Add(b); charList.Add(256 + n); n++; }
        }
        var mapping = new Dictionary<char, byte>(256);
        for (int i = 0; i < 256; i++)
            mapping[(char)charList[i]] = (byte)byteList[i];
        return mapping;
    }

    /// <summary>
    /// Converts a BPE token string (GPT-2 bytes_to_unicode encoded) to raw bytes.
    /// Each character in the token maps to exactly one byte via the inverse mapping.
    /// </summary>
    private byte[] TokenToBytes(string tokenStr)
    {
        var bytes = new byte[tokenStr.Length];
        for (int i = 0; i < tokenStr.Length; i++)
            bytes[i] = _charToByte.TryGetValue(tokenStr[i], out var b) ? b : (byte)tokenStr[i];
        return bytes;
    }

    /// <summary>
    /// Builds a mapping from raw bytes to token IDs using the GPT-2 bytes_to_unicode scheme.
    /// This is needed for BPE/tiktoken tokenizers where control characters (like \n, \r, \t)
    /// and other bytes are represented as unicode characters in the vocabulary.
    /// </summary>
    private Dictionary<byte, int> BuildByteToTokenIdMapping()
    {
        var mapping = new Dictionary<byte, int>();

        // GPT-2 bytes_to_unicode mapping:
        // - Bytes 33-126 ('!' to '~') and 161-172, 174-255 map to themselves
        // - All other bytes (0-32, 127-160, 173) map to unicode chars starting from 256
        var byteList = new List<int>();
        var charList = new List<int>();

        // Printable ASCII: 33 ('!') to 126 ('~')
        for (int i = 33; i <= 126; i++)
        {
            byteList.Add(i);
            charList.Add(i);
        }

        // Extended ASCII: 161 ('¡') to 172 ('¬')
        for (int i = 161; i <= 172; i++)
        {
            byteList.Add(i);
            charList.Add(i);
        }

        // Extended ASCII: 174 ('®') to 255 ('ÿ')
        for (int i = 174; i <= 255; i++)
        {
            byteList.Add(i);
            charList.Add(i);
        }

        // Remaining bytes (0-32, 127-160, 173) get mapped to unicode chars 256+
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!byteList.Contains(b))
            {
                byteList.Add(b);
                charList.Add(256 + n);
                n++;
            }
        }

        // Now build the reverse mapping: for each byte, find the token id
        for (int i = 0; i < 256; i++)
        {
            byte byteVal = (byte)byteList[i];
            char mappedChar = (char)charList[i];
            string charStr = mappedChar.ToString();

            if (_tokenToId.TryGetValue(charStr, out int tokenId))
            {
                mapping[byteVal] = tokenId;
            }
        }

        return mapping;
    }


    /// <summary>
    /// Load tokenizer from GGUF model metadata
    /// </summary>
    public static BPETokenizer FromGGUF(GGUF.GGUFReader reader)
    {
        // Extract vocabulary from GGUF metadata
        if (!reader.Metadata.TryGetValue<string[]>("tokenizer.ggml.tokens", out var tokens) || tokens == null)
            throw new InvalidOperationException("GGUF model does not contain tokenizer.ggml.tokens");

        // Extract special tokens
        int bosToken = 1; // Default
        int eosToken = 2;
        int unknownToken = 0;

        if (reader.Metadata.TryGetValue<int>("tokenizer.ggml.bos_token_id", out var bos))
            bosToken = bos;
        if (reader.Metadata.TryGetValue<int>("tokenizer.ggml.eos_token_id", out var eos))
            eosToken = eos;
        if (reader.Metadata.TryGetValue<int>("tokenizer.ggml.unknown_token_id", out var unk))
            unknownToken = unk;

        // Determine tokenizer type: "gpt2" = BPE/tiktoken (no ▁ prefix), "llama" or "sentencepiece" = SentencePiece
        bool isSentencePiece = true;
        if (reader.Metadata.TryGetValue<string>("tokenizer.ggml.model", out var modelType))
        {
            isSentencePiece = modelType.Equals("llama", StringComparison.OrdinalIgnoreCase)
                           || modelType.Equals("sentencepiece", StringComparison.OrdinalIgnoreCase);
            _logger.LogInformation($"Tokenizer model type: '{modelType}' → isSentencePiece={isSentencePiece}");
        }

        _logger.LogInformation($"Loaded tokenizer: {tokens.Length} tokens, BOS={bosToken}, EOS={eosToken}, model={modelType ?? "unknown"}");

        return new BPETokenizer(tokens, bosToken, eosToken, unknownToken, isSentencePiece);
    }


    /// <summary>
    /// Encode text to token IDs (simplified greedy matching).
    /// Handles special tokens (e.g. <|system|>, <|im_start|>, </s>)
    /// by direct vocab lookup BEFORE SentencePiece normalisation, so they are
    /// never corrupted by the ▁ prefix.
    /// </summary>
    public int[] Encode(string text, bool addBos = true, bool addEos = false)
    {
        var tokens = new List<int>();

        if (addBos)
            tokens.Add(_bosToken);

        // ---- Phase 1: extract known special tokens (angle-bracket tokens) ----
        // Split the text into segments: special tokens (exact vocab matches) and
        // plain text that needs SentencePiece BPE tokenisation.
        var segments = new List<(string text, bool isSpecial)>();
        int scanPos = 0;
        while (scanPos < text.Length)
        {
            // Look for the next '<' character
            int angleStart = text.IndexOf('<', scanPos);
            if (angleStart < 0)
            {
                // No more special tokens — rest is plain text
                segments.Add((text[scanPos..], false));
                break;
            }

            // Plain text before the '<'
            if (angleStart > scanPos)
                segments.Add((text[scanPos..angleStart], false));

            // Find the closing '>'
            int angleEnd = text.IndexOf('>', angleStart + 1);
            if (angleEnd < 0)
            {
                // Unclosed '<' — treat as plain text
                segments.Add((text[scanPos..], false));
                break;
            }

            // Candidate special token (including the angle brackets)
            string candidate = text[angleStart..(angleEnd + 1)];

            // Check if it exists in the vocabulary
            if (_tokenToId.ContainsKey(candidate))
            {
                segments.Add((candidate, true));
                scanPos = angleEnd + 1;
            }
            else
            {
                // Not a known special token — treat '<' as plain text
                segments.Add(("<", false));
                scanPos = angleStart + 1;
            }
        }

        // ---- Phase 2: tokenise each segment ----
        foreach (var (segment, isSpecial) in segments)
        {
            if (isSpecial)
            {
                // Direct lookup — no SentencePiece normalisation
                tokens.Add(_tokenToId[segment]);
            }
            else if (_isSentencePiece)
            {
                // SentencePiece BPE tokenisation with ▁ prefix
                // Used by LLaMA, Mistral, TinyLlama, etc.
                string normalized = "▁" + segment.Replace(" ", "▁");
                int pos = 0;
                while (pos < normalized.Length)
                {
                    int bestMatchLen = 0;
                    int bestTokenId = _unknownToken;

                    for (int len = Math.Min(normalized.Length - pos, 50); len > 0; len--)
                    {
                        string substr = normalized.Substring(pos, len);
                        if (_tokenToId.TryGetValue(substr, out int tokenId))
                        {
                            bestMatchLen = len;
                            bestTokenId = tokenId;
                            break;
                        }
                    }

                    if (bestMatchLen > 0)
                    {
                        tokens.Add(bestTokenId);
                        pos += bestMatchLen;
                    }
                    else
                    {
                        char c = normalized[pos];
                        string charStr = c.ToString();
                        if (_tokenToId.TryGetValue(charStr, out int charTokenId))
                            tokens.Add(charTokenId);
                        else
                            tokens.Add(_unknownToken);
                        pos++;
                    }
                }
            }
            else
            {
                // BPE/tiktoken style (Qwen, GPT-2, etc.)
                // No ▁ prefix, no space replacement.
                // Tokens may contain spaces as part of the token (e.g. " Hello").
                // Use greedy longest-match directly on the raw text.
                // When a character is not found in the vocabulary, use byte-level
                // encoding (GPT-2 bytes_to_unicode mapping) instead of unknown token.
                int pos = 0;
                while (pos < segment.Length)
                {
                    int bestMatchLen = 0;
                    int bestTokenId = _unknownToken;

                    for (int len = Math.Min(segment.Length - pos, 50); len > 0; len--)
                    {
                        string substr = segment.Substring(pos, len);
                        if (_tokenToId.TryGetValue(substr, out int tokenId))
                        {
                            bestMatchLen = len;
                            bestTokenId = tokenId;
                            break;
                        }
                    }

                    if (bestMatchLen > 0)
                    {
                        tokens.Add(bestTokenId);
                        pos += bestMatchLen;
                    }
                    else
                    {
                        // Character not found in vocabulary.
                        // Use byte-level encoding: encode as UTF-8 bytes and map
                        // each byte through the GPT-2 bytes_to_unicode mapping.
                        char c = segment[pos];
                        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(new[] { c });
                        foreach (byte b in bytes)
                        {
                            if (_byteToTokenId.TryGetValue(b, out int byteTokenId))
                                tokens.Add(byteTokenId);
                            else
                                tokens.Add(_unknownToken);
                        }
                        pos++;
                    }
                }
            }
        }


        if (addEos)
            tokens.Add(_eosToken);

        return tokens.ToArray();
    }

    /// <summary>
    /// Decode token IDs to text
    /// </summary>
    public string Decode(int[] tokenIds)
    {
        if (_isSentencePiece)
        {
            var result = new System.Text.StringBuilder();
            foreach (int tokenId in tokenIds)
            {
                if (tokenId >= 0 && tokenId < _tokens.Length)
                {
                    if (tokenId == _bosToken || tokenId == _eosToken) continue;
                    result.Append(_tokens[tokenId].Replace('▁', ' '));
                }
            }
            return result.Length > 0 && result[0] == ' '
                ? result.ToString(1, result.Length - 1)
                : result.ToString();
        }
        else
        {
            // BPE/tiktoken: each token is GPT-2 bytes_to_unicode encoded.
            // Accumulate all raw bytes across tokens, then UTF-8 decode the whole buffer.
            var allBytes = new System.Collections.Generic.List<byte>();
            foreach (int tokenId in tokenIds)
            {
                if (tokenId >= 0 && tokenId < _tokens.Length)
                {
                    if (tokenId == _bosToken || tokenId == _eosToken) continue;
                    allBytes.AddRange(TokenToBytes(_tokens[tokenId]));
                }
            }
            return System.Text.Encoding.UTF8.GetString(allBytes.ToArray());
        }
    }

    /// <summary>
    /// Returns the raw bytes represented by this token (for BPE/tiktoken via inverse
    /// bytes_to_unicode mapping, for SentencePiece via UTF-8 encoding of the ▁-replaced text).
    /// Used by streaming engines to buffer partial UTF-8 sequences safely.
    /// </summary>
    public byte[] GetTokenBytes(int tokenId)
    {
        if (tokenId < 0 || tokenId >= _tokens.Length) return [];
        if (tokenId == _bosToken || tokenId == _eosToken) return [];
        if (_isSentencePiece)
            return System.Text.Encoding.UTF8.GetBytes(_tokens[tokenId].Replace('▁', ' '));
        return TokenToBytes(_tokens[tokenId]);
    }

    /// <summary>
    /// Get token text by ID. For BPE/tiktoken, decodes bytes_to_unicode back to UTF-8.
    /// Note: streaming may emit incomplete UTF-8 sequences for split multi-byte characters;
    /// use Decode() on the full token sequence for guaranteed correctness.
    /// </summary>
    public string GetToken(int tokenId)
    {
        if (tokenId >= 0 && tokenId < _tokens.Length)
        {
            if (_isSentencePiece)
                return _tokens[tokenId].Replace('▁', ' ');
            // BPE: decode via inverse bytes_to_unicode, then interpret as UTF-8
            return System.Text.Encoding.UTF8.GetString(TokenToBytes(_tokens[tokenId]));
        }
        return "<unk>";
    }


    /// <summary>
    /// Direct vocabulary lookup — returns the token ID for an exact vocab entry,
    /// or -1 if not found. Use this for special tokens like &lt;|im_end|&gt; that
    /// must NOT go through SentencePiece normalization.
    /// </summary>
    public int GetTokenId(string vocabEntry)
        => _tokenToId.TryGetValue(vocabEntry, out var id) ? id : -1;

    /// <summary>
    /// Encode stop sequences properly: tries direct vocab lookup first (for special
    /// tokens like &lt;|im_end|&gt;), then falls back to regular BPE encoding.
    /// </summary>
    public int[] EncodeStopSequence(string text)
    {
        // Direct lookup for exact special-token strings (no normalization)
        if (_tokenToId.TryGetValue(text, out var specialId))
            return [specialId];
        // Fallback: standard BPE without BOS
        return Encode(text, addBos: false, addEos: false);
    }
}
