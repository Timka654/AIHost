#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer TokenIds { int ids[]; } tokenIds;
layout(set = 0, binding = 1) readonly buffer EmbTable { float data[]; } table;
layout(set = 0, binding = 2) writeonly buffer Output { float data[]; } output_buf;
layout(set = 0, binding = 3) readonly buffer Params { uint seqLen; uint dModel; } params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= params.seqLen * params.dModel) return;
    
    uint seq = gid / params.dModel;
    uint dim = gid % params.dModel;
    
    int tokenId = tokenIds.ids[seq];
    if (tokenId < 0) return;
    
    output_buf.data[gid] = table.data[uint(tokenId) * params.dModel + dim];
}
