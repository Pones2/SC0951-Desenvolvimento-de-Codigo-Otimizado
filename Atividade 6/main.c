#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define W 1920  // largura da imagem
#define H 1080   // altura da imagem

typedef float pixel_t;


double wall_time() {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

// blur 1D horizontal escalar: média de 3 pixels (x-1,x,x+1)
void blur_horizontal_scalar(const pixel_t *in, pixel_t *out, int w, int h) {
    const float factor = 1.0f / 3.0f;

    for (int y = 0; y < h; y++) {
        // borda esquerda: copia
        out[y * w + 0] = in[y * w + 0];

        for (int x = 1; x < w - 1; x++) {
            float left   = in[y * w + (x - 1)];
            float center = in[y * w + x];
            float right  = in[y * w + (x + 1)];
            out[y * w + x] = (left + center + right) * factor;
        }

        // borda direita: copia
        out[y * w + (w - 1)] = in[y * w + (w - 1)];
    }
}

// blur 1D vertical escalar: média de 3 pixels (y-1,y,y+1)
void blur_vertical_scalar(const pixel_t *in, pixel_t *out, int w, int h) {
    const float factor = 1.0f / 3.0f;

    // primeira linha: copia
    for (int x = 0; x < w; x++) {
        out[0 * w + x] = in[0 * w + x];
    }

    for (int y = 1; y < h - 1; y++) {
        for (int x = 0; x < w; x++) {
            float up    = in[(y - 1) * w + x];
            float center= in[y * w + x];
            float down  = in[(y + 1) * w + x];
            out[y * w + x] = (up + center + down) * factor;
        }
    }

    // última linha: copia
    for (int x = 0; x < w; x++) {
        out[(h - 1) * w + x] = in[(h - 1) * w + x];
    }
}

// blur completo escalar: horizontal + vertical
void blur_scalar(const pixel_t *in, pixel_t *out, int w, int h) {
    pixel_t *tmp = (pixel_t*)malloc(w * h * sizeof(pixel_t));
    if (!tmp) {
        fprintf(stderr, "Erro ao alocar tmp\n");
        exit(1);
    }

    blur_horizontal_scalar(in, tmp, w, h);
    blur_vertical_scalar(tmp, out, w, h);

    free(tmp);
}

// blur 1D horizontal com AVX: processa 8 pixels por vez
void blur_horizontal_avx(const pixel_t *in, pixel_t *out, int w, int h) {
    const float factor_val = 1.0f / 3.0f;
    __m256 factor = _mm256_set1_ps(factor_val);

    for (int y = 0; y < h; y++) {
        int row_offset = y * w;

        // borda esquerda
        out[row_offset + 0] = in[row_offset + 0];

        int x = 1;

        // processa blocos de 8 pixels, garantindo que x+7+1 < w  => x <= w-9
        int limit = w - 1 - 8;  // último x onde x+7 ainda tem vizinho à direita
        for (; x <= limit; x += 8) {
            // carrega [x-1 .. x+6]
            __m256 left   = _mm256_loadu_ps(&in[row_offset + x - 1]);
            // carrega [x   .. x+7]
            __m256 center = _mm256_loadu_ps(&in[row_offset + x]);
            // carrega [x+1 .. x+8]
            __m256 right  = _mm256_loadu_ps(&in[row_offset + x + 1]);

            __m256 sum = _mm256_add_ps(left, center);
            sum = _mm256_add_ps(sum, right);
            __m256 res = _mm256_mul_ps(sum, factor);

            _mm256_storeu_ps(&out[row_offset + x], res);
        }

        // resto (até w-2) faz escalar
        for (; x < w - 1; x++) {
            float left   = in[row_offset + (x - 1)];
            float center = in[row_offset + x];
            float right  = in[row_offset + (x + 1)];
            out[row_offset + x] = (left + center + right) * factor_val;
        }

        // borda direita
        out[row_offset + (w - 1)] = in[row_offset + (w - 1)];
    }
}

// blur completo vetorizado: horizontal (AVX) + vertical (escalar)
void blur_avx(const pixel_t *in, pixel_t *out, int w, int h) {
    pixel_t *tmp = (pixel_t*)malloc(w * h * sizeof(pixel_t));
    if (!tmp) {
        fprintf(stderr, "Erro ao alocar tmp\n");
        exit(1);
    }

    blur_horizontal_avx(in, tmp, w, h);
    blur_vertical_scalar(tmp, out, w, h);

}

// ---------- Função para comparar resultados ----------
float max_abs_diff(const pixel_t *a, const pixel_t *b, int w, int h) {
    float maxd = 0.0f;
    int n = w * h;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        if (d < 0) d = -d;
        if (d > maxd) maxd = d;
    }
    return maxd;
}

// ---------- MAIN ----------
int main(void) {
    // aloca imagens
    pixel_t *img_in      = (pixel_t*)malloc(W * H * sizeof(pixel_t));
    pixel_t *img_scalar  = (pixel_t*)malloc(W * H * sizeof(pixel_t));
    pixel_t *img_avx     = (pixel_t*)malloc(W * H * sizeof(pixel_t));

    if (!img_in || !img_scalar || !img_avx) {
        fprintf(stderr, "Erro ao alocar memoria\n");
        return 1;
    }

    // inicializa imagem de entrada com valores aleatórios [0,1]
    srand(123);
    for (int i = 0; i < W * H; i++) {
        img_in[i] = (float)rand() / (float)RAND_MAX;
    }

    // BLUR ESCALAR 100X
    double t0 = wall_time();
    for (int r = 0; r < 100; r++)
        blur_scalar(img_in, img_scalar, W, H);
    double t1 = wall_time();
    double tempo_scalar = t1 - t0;

    // BLUR AVX 100X
    double t2 = wall_time();
    for (int r = 0; r < 100; r++)
        blur_avx(img_in, img_avx, W, H);
    double t3 = wall_time();
    double tempo_avx = t3 - t2;

    // compara resultados
    float diff = max_abs_diff(img_scalar, img_avx, W, H);

    printf("Tempo ESCALAR:   %.6f s\n", tempo_scalar);
    printf("Tempo AVX:       %.6f s\n", tempo_avx);
    printf("Speedup (esc/avx): %.2fx\n", tempo_scalar / tempo_avx);
    printf("Max diff entre resultados: %g\n", diff);


    free(img_in);
    free(img_scalar);
    free(img_avx);

    return 0;
}
