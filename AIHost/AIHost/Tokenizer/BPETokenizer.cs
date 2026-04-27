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

    public int VocabSize => _tokens.Length;
    public int BosToken => _bosToken;
    public int EosToken => _eosToken;

    public BPETokenizer(string[] tokens, int bosToken = 1, int eosToken = 2, int unknownToken = 0)
    {
        _tokens = tokens;
        _bosToken = bosToken;
        _eosToken = eosToken;
        _unknownToken = unknownToken;

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

        Console.WriteLine($"Loaded tokenizer: {tokens.Length} tokens, BOS={bosToken}, EOS={eosToken}");

        return new BPETokenizer(tokens, bosToken, eosToken, unknownToken);
    }

    /// <summary>
    /// Encode text to token IDs (simplified greedy matching)
    /// </summary>
    public int[] Encode(string text, bool addBos = true, bool addEos = false)
    {
        var tokens = new List<int>();

        if (addBos)
            tokens.Add(_bosToken);

        // SentencePiece convention: spaces are represented as ▁ prefix on tokens.
        // Prepend ▁ and replace all spaces so greedy matching finds real vocab entries.
        string normalized = "▁" + text.Replace(" ", "▁");

        // Simple greedy tokenization: try longest match first
        int pos = 0;
        while (pos < normalized.Length)
        {
            int bestMatchLen = 0;
            int bestTokenId = _unknownToken;

            // Try to find longest matching token
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
                // Fallback: try to find single byte token
                char c = normalized[pos];
                string charStr = c.ToString();
                if (_tokenToId.TryGetValue(charStr, out int charTokenId))
                {
                    tokens.Add(charTokenId);
                }
                else
                {
                    tokens.Add(_unknownToken);
                }
                pos++;
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

                // SentencePiece: ▁ marks word boundary (space before word)
                result.Append(_tokens[tokenId].Replace('▁', ' '));
            }
        }

        // Leading space is an artifact of the ▁ prepended during encoding
        return result.Length > 0 && result[0] == ' '
            ? result.ToString(1, result.Length - 1)
            : result.ToString();
    }

    /// <summary>
    /// Get token text by ID
    /// </summary>
    public string GetToken(int tokenId)
    {
        if (tokenId >= 0 && tokenId < _tokens.Length)
            return _tokens[tokenId].Replace('▁', ' ');
        return "<unk>";
    }
}
