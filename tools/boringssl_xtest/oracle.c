// BoringSSL ML-KEM-768 interop oracle.
//
// A tiny stdin/stdout line-protocol oracle in the style of sqisign-selkie's
// cref_xtest oracle, but talking to BoringSSL's ML-KEM-768 instead of a NIST C
// reference. The Rust test (tests/boringssl_xtest.rs) spawns this binary and
// cross-checks every direction against mlkem-selkie.
//
// Protocol (hex, one case per line):
//   in:  <seed_hex(64 bytes)> <our_ct_hex(1088 bytes)>
//   out: <ek_hex(1184)> <bssl_ct_hex(1088)> <bssl_ss_hex(32)> <our_ss_hex(32)>
//        or the literal "FAIL" if BoringSSL rejects the seed.
//
// Per line the oracle: derives the key pair from the seed (deterministic),
// marshals the encapsulation key, encapsulates under it (BoringSSL's own
// randomness), and decapsulates the caller's ciphertext. That lets the Rust
// side check, from one round trip: keygen byte-identity (ek), their-encaps /
// our-decaps (bssl_ct, bssl_ss), and our-encaps / their-decaps (our_ss).
//
// Build with tools/boringssl_xtest/build.sh against a prebuilt BoringSSL.

#include <openssl/bytestring.h>
#include <openssl/mlkem.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int hex_to_bytes(const char *hex, size_t hex_len, uint8_t *out, size_t out_len) {
    if (hex_len != out_len * 2) {
        return 0;
    }
    for (size_t i = 0; i < out_len; i++) {
        unsigned int byte;
        if (sscanf(hex + 2 * i, "%2x", &byte) != 1) {
            return 0;
        }
        out[i] = (uint8_t)byte;
    }
    return 1;
}

static void print_hex(const uint8_t *bytes, size_t len) {
    for (size_t i = 0; i < len; i++) {
        printf("%02x", bytes[i]);
    }
}

int main(void) {
    char *line = NULL;
    size_t cap = 0;
    ssize_t len;

    while ((len = getline(&line, &cap, stdin)) > 0) {
        // Trim the trailing newline.
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) {
            continue;
        }

        char *space = strchr(line, ' ');
        if (space == NULL) {
            printf("FAIL\n");
            fflush(stdout);
            continue;
        }
        *space = '\0';
        const char *seed_hex = line;
        const char *our_ct_hex = space + 1;

        uint8_t seed[MLKEM_SEED_BYTES];
        uint8_t our_ct[MLKEM768_CIPHERTEXT_BYTES];
        if (!hex_to_bytes(seed_hex, strlen(seed_hex), seed, sizeof seed) ||
            !hex_to_bytes(our_ct_hex, strlen(our_ct_hex), our_ct, sizeof our_ct)) {
            printf("FAIL\n");
            fflush(stdout);
            continue;
        }

        struct MLKEM768_private_key priv;
        if (!MLKEM768_private_key_from_seed(&priv, seed, sizeof seed)) {
            printf("FAIL\n");
            fflush(stdout);
            continue;
        }

        struct MLKEM768_public_key pub;
        MLKEM768_public_from_private(&pub, &priv);

        uint8_t ek[MLKEM768_PUBLIC_KEY_BYTES];
        CBB cbb;
        if (!CBB_init_fixed(&cbb, ek, sizeof ek) || !MLKEM768_marshal_public_key(&cbb, &pub) ||
            CBB_len(&cbb) != sizeof ek) {
            printf("FAIL\n");
            fflush(stdout);
            continue;
        }

        uint8_t bssl_ct[MLKEM768_CIPHERTEXT_BYTES];
        uint8_t bssl_ss[MLKEM_SHARED_SECRET_BYTES];
        MLKEM768_encap(bssl_ct, bssl_ss, &pub);

        uint8_t our_ss[MLKEM_SHARED_SECRET_BYTES];
        if (!MLKEM768_decap(our_ss, our_ct, sizeof our_ct, &priv)) {
            printf("FAIL\n");
            fflush(stdout);
            continue;
        }

        print_hex(ek, sizeof ek);
        putchar(' ');
        print_hex(bssl_ct, sizeof bssl_ct);
        putchar(' ');
        print_hex(bssl_ss, sizeof bssl_ss);
        putchar(' ');
        print_hex(our_ss, sizeof our_ss);
        putchar('\n');
        fflush(stdout);
    }

    free(line);
    return 0;
}
