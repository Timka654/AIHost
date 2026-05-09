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
            Console.WriteLine($"Tokenizer model type: '{modelType}' → isSentencePiece={isSentencePiece}");
        }

        Console.WriteLine($"Loaded tokenizer: {tokens.Length} tokens, BOS={bosToken}, EOS={eosToken}, model={modelType ?? "unknown"}");

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
                        // Try single character
                        char c = segment[pos];
                        string charStr = c.ToString();
                        if (_tokenToId.TryGetValue(charStr, out int charTokenId))
                            tokens.Add(charTokenId);
                        else
                            tokens.Add(_unknownToken);
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
        var result = new System.Text.StringBuilder();

        foreach (int tokenId in tokenIds)
        {
            if (tokenId >= 0 && tokenId < _tokens.Length)
            {
                if (tokenId == _bosToken || tokenId == _eosToken)
                    continue;

                if (_isSentencePiece)
                {
                    // SentencePiece: ▁ marks word boundary (space before word)
                    result.Append(_tokens[tokenId].Replace('▁', ' '));
                }
                else
                {
                    // BPE/tiktoken: tokens are raw text (may contain spaces)
                    result.Append(_tokens[tokenId]);
                }
            }
        }

        if (_isSentencePiece)
        {
            // Leading space is an artifact of the ▁ prepended during encoding
            return result.Length > 0 && result[0] == ' '
                ? result.ToString(1, result.Length - 1)
                : result.ToString();
        }
        return result.ToString();
    }

    /// <summary>
    /// Get token text by ID
    /// </summary>
    public string GetToken(int tokenId)
    {
        if (tokenId >= 0 && tokenId < _tokens.Length)
        {
            if (_isSentencePiece)
                return _tokens[tokenId].Replace('▁', ' ');
            return _tokens[tokenId];
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
