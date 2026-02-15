pragma circom 2.0.0;

include "../node_modules/circomlib/circuits/poseidon.circom";

template DnaMatch() {
    signal input private_dna_array[20];
    signal input salt;
    signal input public_hash;

    // chunk hashing
    component chunkHashes[4];

    for (var i = 0; i < 4; i++) {
        chunkHashes[i] = Poseidon(5);
        for (var j = 0; j < 5; j++) {
            chunkHashes[i].inputs[j] <== private_dna_array[i * 5 + j];
        }
    }

    // final hash
    component finalHasher = Poseidon(5);
    finalHasher.inputs[0] <== chunkHashes[0].out;
    finalHasher.inputs[1] <== chunkHashes[1].out;
    finalHasher.inputs[2] <== chunkHashes[2].out;
    finalHasher.inputs[3] <== chunkHashes[3].out;
    finalHasher.inputs[4] <== salt;

    finalHasher.out === public_hash;
}

component main {public [public_hash]} = DnaMatch();
