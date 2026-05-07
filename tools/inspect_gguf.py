#!/usr/bin/env python3
"""Inspect a GGUF file: print architecture metadata and tensor names.
Usage: python inspect_gguf.py path/to/model.gguf
"""
import sys, struct, json

def read_gguf_info(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', f"Not a GGUF file: {magic}"
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        kv_count     = struct.unpack('<Q', f.read(8))[0]

        print(f"GGUF version:    {version}")
        print(f"Tensor count:    {tensor_count}")
        print(f"Metadata pairs:  {kv_count}\n")

        # --- read metadata KV ---
        TYPES = {0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',
                 8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}
        SIZES = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

        def read_str():
            n = struct.unpack('<Q', f.read(8))[0]
            return f.read(n).decode('utf-8', errors='replace')

        def read_val(t):
            if t == 8:   return read_str()
            if t == 7:   return bool(struct.unpack('B', f.read(1))[0])
            if t in SIZES:
                fmt = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',10:'Q',11:'q',12:'d'}[t]
                return struct.unpack('<'+fmt, f.read(SIZES[t]))[0]
            if t == 9:   # array
                elem_t = struct.unpack('<I', f.read(4))[0]
                n      = struct.unpack('<Q', f.read(8))[0]
                return [read_val(elem_t) for _ in range(min(n, 8))]  # cap at 8
            return None

        meta = {}
        for _ in range(kv_count):
            key   = read_str()
            vtype = struct.unpack('<I', f.read(4))[0]
            val   = read_val(vtype)
            meta[key] = val

        arch = meta.get('general.architecture', '?')
        print("=== Key metadata ===")
        important = ['general.architecture','general.name','general.quantization_version',
                     f'{arch}.block_count',f'{arch}.embedding_length',
                     f'{arch}.attention.head_count',f'{arch}.attention.head_count_kv',
                     f'{arch}.feed_forward_length',f'{arch}.expert_count',
                     f'{arch}.expert_used_count',f'{arch}.rope.freq_base']
        for k in important:
            if k in meta: print(f"  {k} = {meta[k]}")
        print()

        # --- read tensor info ---
        print("=== Tensor names (first 60) ===")
        tensor_names = []
        TTYPE_NAMES = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',
                       8:'Q8_0',9:'Q8_1',10:'Q2_K',11:'Q3_K',12:'Q4_K',
                       13:'Q5_K',14:'Q6_K',15:'Q8_K',16:'IQ2_XXS',17:'IQ2_XS',
                       18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',
                       22:'IQ2_S',23:'IQ4_XS',24:'I8',25:'I16',26:'I32',
                       27:'I64',28:'F64',29:'IQ1_M',30:'BF16',31:'Q4_0_4_4',
                       32:'Q4_0_4_8',33:'Q4_0_8_8',34:'TQ1_0',35:'TQ2_0'}
        for _ in range(tensor_count):
            name  = read_str()
            ndims = struct.unpack('<I', f.read(4))[0]
            dims  = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            _offset = struct.unpack('<Q', f.read(8))[0]
            tname = TTYPE_NAMES.get(ttype, str(ttype))
            tensor_names.append((name, tname, dims))
            if _ < 60:
                print(f"  {name:<55} {tname:<8} {dims}")

        if tensor_count > 60:
            print(f"  ... ({tensor_count - 60} more tensors)")

        # --- summary: unique name patterns ---
        print("\n=== Unique name patterns (blk.0.*) ===")
        blk0 = [n for n,_,_ in tensor_names if n.startswith('blk.0.')]
        for n in blk0:
            print(f"  {n}")

        print("\n=== Architecture detection hint ===")
        print(f"  general.architecture = '{arch}'")
        has_q = any('attn_q.' in n for n,_,_ in tensor_names)
        has_qkv = any('attn_qkv' in n for n,_,_ in tensor_names)
        has_moe = any('ffn_gate_exps' in n or 'ffn_exp.' in n for n,_,_ in tensor_names)
        has_mla = any('attn_q_a' in n for n,_,_ in tensor_names)
        print(f"  Separate Q/K/V:   {has_q}")
        print(f"  Combined QKV:     {has_qkv}")
        print(f"  MoE FFN:          {has_moe}")
        print(f"  MLA attention:    {has_mla}")
        return meta, tensor_names

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_gguf.py /path/to/model.gguf"); sys.exit(1)
    read_gguf_info(sys.argv[1])
