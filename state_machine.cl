__constant uint K[] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Helper functions
uint Ch(uint x, uint y, uint z) { return (x & y) ^ (~x & z); }
uint Maj(uint x, uint y, uint z) { return (x & y) ^ (x & z) ^ (y & z); }
uint rotr(uint x, uint n) { return (x >> n) | (x << (32 - n)); }
uint Sigma0(uint x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
uint Sigma1(uint x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
uint sigma0(uint x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
uint sigma1(uint x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

// Function to perform SHA-256 on the internal state
void sha256_transform(__global uint *state, __private uchar *data) {
    uint W[64];
    uint a, b, c, d, e, f, g, h, temp1, temp2;

    // Initialize working variables
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    // Prepare message schedule W
    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint)data[i * 4] << 24) | ((uint)data[i * 4 + 1] << 16) | ((uint)data[i * 4 + 2] << 8) | ((uint)data[i * 4 + 3]);
    }

    for (int i = 16; i < 64; ++i) {
        W[i] = sigma1(W[i - 2]) + W[i - 7] + sigma0(W[i - 15]) + W[i - 16];
    }

    // Perform the main SHA-256 loop
    for (int i = 0; i < 64; ++i) {
        temp1 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i];
        temp2 = Sigma0(a) + Maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    // Update state
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__kernel void crng_kernel(__global uint *state, __global uchar *output, __global uchar *entropy_input, __global ulong *time_seed, int input_length) {
    int gid = get_global_id(0);  // Global work-item ID
    int lid = get_local_id(0);   // Local work-item ID

    // Combine time-based seed with local and global IDs to add more entropy
    ulong local_entropy = time_seed[gid % get_global_size(0)] ^ (ulong)get_global_id(0) ^ (ulong)get_local_id(0);

    // Use local_entropy to seed the state with more entropy
    if (state[0] == 0 && state[1] == 0) {
        for (int i = 0; i < 8; i++) {
            state[i] = ((uint)entropy_input[i * 4] << 24) |
                       ((uint)entropy_input[i * 4 + 1] << 16) |
                       ((uint)entropy_input[i * 4 + 2] << 8) |
                       ((uint)entropy_input[i * 4 + 3]);
        }
        // XOR local_entropy into the state to introduce time-based variability
        state[0] ^= (uint)(local_entropy >> 32);  // Use upper 32 bits
        state[1] ^= (uint)(local_entropy & 0xFFFFFFFF);  // Use lower 32 bits
    }

    // Mix additional entropy into the state (reseed step)
    for (int i = 0; i < input_length; i++) {
        state[i % 8] ^= (uint)entropy_input[i];
    }

    // Generate random data using SHA-256 on the state
    uchar temp_data[64]; // Temporary buffer for hash
    for (int i = 0; i < 8; i++) {
        temp_data[i * 4] = (state[i] >> 24) & 0xFF;
        temp_data[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        temp_data[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        temp_data[i * 4 + 3] = state[i] & 0xFF;
    }

    // Perform SHA-256 on state
    sha256_transform(state, temp_data);

    // Feedback loop: mix the first 16 bytes of the random output back into the state
    for (int i = 0; i < 8; i++) {
        state[i] ^= ((uint)temp_data[i * 2] << 24) |
                    ((uint)temp_data[i * 2 + 1] << 16);
    }

    // Output random bytes (generated from state)
    for (int i = 0; i < 32; i++) {
        output[gid * 32 + i] = temp_data[i];
    }
}
