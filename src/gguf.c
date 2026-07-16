#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* ---- little-endian readers from a cursor ---- */

typedef struct {
    const uint8_t *p;
    const uint8_t *end;
} Cursor;

static int need(Cursor *c, size_t n) {
    return (size_t)(c->end - c->p) >= n;
}

static uint8_t  r_u8(Cursor *c)  { uint8_t v = *c->p++; return v; }
static uint16_t r_u16(Cursor *c) {
    uint16_t v; memcpy(&v, c->p, 2); c->p += 2; return v;
}
static uint32_t r_u32(Cursor *c) {
    uint32_t v; memcpy(&v, c->p, 4); c->p += 4; return v;
}
static uint64_t r_u64(Cursor *c) {
    uint64_t v; memcpy(&v, c->p, 8); c->p += 8; return v;
}
static int32_t  r_i32(Cursor *c) { return (int32_t)r_u32(c); }
static float    r_f32(Cursor *c) {
    float v; memcpy(&v, c->p, 4); c->p += 4; return v;
}
static double   r_f64(Cursor *c) {
    double v; memcpy(&v, c->p, 8); c->p += 8; return v;
}

static char *r_string(Cursor *c) {
    if (!need(c, 8)) return NULL;
    uint64_t n = r_u64(c);
    if (!need(c, (size_t)n)) return NULL;
    char *s = (char *)malloc((size_t)n + 1);
    if (!s) return NULL;
    memcpy(s, c->p, (size_t)n);
    s[n] = '\0';
    c->p += n;
    return s;
}

static size_t type_scalar_size(int t) {
    switch (t) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:   return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:  return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static int skip_value(Cursor *c, int type);

static int read_value(Cursor *c, int type, GGUFKV *kv) {
    kv->type = type;
    kv->arr_type = -1;
    kv->arr_len = 0;
    kv->v.arr = NULL;
    kv->v.str = NULL;
    switch (type) {
        case GGUF_TYPE_UINT8:   kv->v.u8  = r_u8(c); break;
        case GGUF_TYPE_INT8:    kv->v.i8  = (int8_t)r_u8(c); break;
        case GGUF_TYPE_UINT16:  kv->v.u16 = r_u16(c); break;
        case GGUF_TYPE_INT16:   kv->v.i16 = (int16_t)r_u16(c); break;
        case GGUF_TYPE_UINT32:  kv->v.u32 = r_u32(c); break;
        case GGUF_TYPE_INT32:   kv->v.i32 = r_i32(c); break;
        case GGUF_TYPE_FLOAT32: kv->v.f32 = r_f32(c); break;
        case GGUF_TYPE_BOOL:    kv->v.b   = r_u8(c); break;
        case GGUF_TYPE_STRING:
            kv->v.str = r_string(c);
            if (!kv->v.str) return -1;
            break;
        case GGUF_TYPE_UINT64:  kv->v.u64 = r_u64(c); break;
        case GGUF_TYPE_INT64:   kv->v.i64 = (int64_t)r_u64(c); break;
        case GGUF_TYPE_FLOAT64: kv->v.f64 = r_f64(c); break;
        case GGUF_TYPE_ARRAY: {
            if (!need(c, 16)) return -1;
            int at = (int)r_u32(c);
            uint64_t n = r_u64(c);
            kv->arr_type = at;
            kv->arr_len = n;
            if (at == GGUF_TYPE_STRING) {
                char **arr = (char **)calloc((size_t)n, sizeof(char *));
                if (!arr) return -1;
                for (uint64_t i = 0; i < n; i++) {
                    arr[i] = r_string(c);
                    if (!arr[i]) { /* leak on error path simplified */ return -1; }
                }
                kv->v.arr = arr;
            } else {
                size_t es = type_scalar_size(at);
                if (es == 0) {
                    /* nested arrays rare in model metadata — skip by recursive read not stored */
                    for (uint64_t i = 0; i < n; i++) {
                        if (skip_value(c, at) != 0) return -1;
                    }
                    kv->v.arr = NULL;
                } else {
                    size_t nbytes = (size_t)n * es;
                    if (!need(c, nbytes)) return -1;
                    void *buf = malloc(nbytes);
                    if (!buf) return -1;
                    memcpy(buf, c->p, nbytes);
                    c->p += nbytes;
                    kv->v.arr = buf;
                }
            }
            break;
        }
        default:
            fprintf(stderr, "initium/gguf: unknown value type %d\n", type);
            return -1;
    }
    return 0;
}

static int skip_value(Cursor *c, int type) {
    GGUFKV tmp;
    memset(&tmp, 0, sizeof(tmp));
    if (read_value(c, type, &tmp) != 0) return -1;
    if (type == GGUF_TYPE_STRING) free(tmp.v.str);
    if (type == GGUF_TYPE_ARRAY) {
        if (tmp.arr_type == GGUF_TYPE_STRING && tmp.v.arr) {
            char **arr = (char **)tmp.v.arr;
            for (uint64_t i = 0; i < tmp.arr_len; i++) free(arr[i]);
        }
        free(tmp.v.arr);
    }
    return 0;
}

static size_t ggml_type_size(int type, const uint64_t ne[4], int n_dims) {
    uint64_t n_elem = 1;
    for (int i = 0; i < n_dims; i++) n_elem *= ne[i];
    switch (type) {
        case GGML_TYPE_F32:  return (size_t)n_elem * 4;
        case GGML_TYPE_F16:  return (size_t)n_elem * 2;
        case GGML_TYPE_Q8_0: return (size_t)(n_elem / 32) * sizeof(uint16_t) /*d*/ + (size_t)n_elem; /* approx block */
        case GGML_TYPE_Q4_0: return (size_t)(n_elem / 32) * (sizeof(uint16_t) + 16);
        default: return 0;
    }
}

/* More precise block sizes matching ggml */
static size_t ggml_nbytes(int type, const uint64_t ne[4], int n_dims) {
    uint64_t ne0 = n_dims > 0 ? ne[0] : 1;
    uint64_t n_blocks;
    switch (type) {
        case GGML_TYPE_F32:
            return (size_t)(ne[0] * (n_dims > 1 ? ne[1] : 1) * (n_dims > 2 ? ne[2] : 1) * (n_dims > 3 ? ne[3] : 1)) * 4;
        case GGML_TYPE_F16:
            return (size_t)(ne[0] * (n_dims > 1 ? ne[1] : 1) * (n_dims > 2 ? ne[2] : 1) * (n_dims > 3 ? ne[3] : 1)) * 2;
        case GGML_TYPE_Q8_0: {
            /* block = 32 weights, 34 bytes (2 scale + 32 i8) */
            uint64_t n_elem = 1;
            for (int i = 0; i < n_dims; i++) n_elem *= ne[i];
            n_blocks = n_elem / 32;
            return (size_t)n_blocks * 34;
        }
        case GGML_TYPE_Q4_0: {
            uint64_t n_elem = 1;
            for (int i = 0; i < n_dims; i++) n_elem *= ne[i];
            n_blocks = n_elem / 32;
            return (size_t)n_blocks * 18;
        }
        case GGML_TYPE_Q6_K: {
            /* block_q6_K = 210 bytes, QK_K = 256 */
            uint64_t n_elem = 1;
            for (int i = 0; i < n_dims; i++) n_elem *= ne[i];
            n_blocks = n_elem / 256;
            return (size_t)n_blocks * 210;
        }
        default:
            (void)ne0;
            return 0;
    }
}

static int map_file(const char *path, GGUFFile *gf) {
#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "initium/gguf: cannot open '%s' (win err %lu)\n", path, GetLastError());
        return -1;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(hFile, &sz)) {
        CloseHandle(hFile);
        return -1;
    }
    gf->map_size = (size_t)sz.QuadPart;
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) {
        CloseHandle(hFile);
        return -1;
    }
    void *view = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        CloseHandle(hMap);
        CloseHandle(hFile);
        return -1;
    }
    gf->map = view;
    gf->win_file = (void *)hFile;
    gf->win_map = (void *)hMap;
    return 0;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("initium/gguf: open");
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return -1;
    }
    gf->map_size = (size_t)st.st_size;
    void *p = mmap(NULL, gf->map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        perror("initium/gguf: mmap");
        return -1;
    }
    gf->map = p;
    return 0;
#endif
}

static void unmap_file(GGUFFile *gf) {
#ifdef _WIN32
    if (gf->map) UnmapViewOfFile(gf->map);
    if (gf->win_map) CloseHandle((HANDLE)gf->win_map);
    if (gf->win_file) CloseHandle((HANDLE)gf->win_file);
    gf->map = gf->win_map = gf->win_file = NULL;
#else
    if (gf->map && gf->map_size) munmap(gf->map, gf->map_size);
    gf->map = NULL;
#endif
}

int gguf_open(const char *path, GGUFFile *gf) {
    memset(gf, 0, sizeof(*gf));
    if (map_file(path, gf) != 0) return -1;
    if (gf->map_size < 24) {
        fprintf(stderr, "initium/gguf: file too small\n");
        gguf_close(gf);
        return -1;
    }

    Cursor c = { (const uint8_t *)gf->map, (const uint8_t *)gf->map + gf->map_size };

    /* magic */
    if (!need(&c, 4) || memcmp(c.p, "GGUF", 4) != 0) {
        fprintf(stderr, "initium/gguf: bad magic (not a GGUF file)\n");
        gguf_close(gf);
        return -1;
    }
    c.p += 4;
    gf->version = (int)r_u32(&c);
    if (gf->version < 2 || gf->version > 3) {
        fprintf(stderr, "initium/gguf: unsupported version %d\n", gf->version);
        gguf_close(gf);
        return -1;
    }

    uint64_t n_tensors = r_u64(&c);
    uint64_t n_kv = r_u64(&c);
    if (n_tensors > 100000 || n_kv > 100000) {
        fprintf(stderr, "initium/gguf: absurd header counts (corrupt?)\n");
        gguf_close(gf);
        return -1;
    }
    gf->n_kv = (int)n_kv;
    gf->n_tensors = (int)n_tensors;
    gf->kv = (GGUFKV *)calloc((size_t)n_kv, sizeof(GGUFKV));
    gf->tensors = (GGUFTensor *)calloc((size_t)n_tensors, sizeof(GGUFTensor));
    if (!gf->kv || !gf->tensors) {
        gguf_close(gf);
        return -1;
    }

    for (uint64_t i = 0; i < n_kv; i++) {
        gf->kv[i].key = r_string(&c);
        if (!gf->kv[i].key) {
            fprintf(stderr, "initium/gguf: truncated KV key at %llu\n", (unsigned long long)i);
            gguf_close(gf);
            return -1;
        }
        if (!need(&c, 4)) { gguf_close(gf); return -1; }
        int vtype = (int)r_u32(&c);
        if (read_value(&c, vtype, &gf->kv[i]) != 0) {
            fprintf(stderr, "initium/gguf: bad value for key '%s'\n", gf->kv[i].key);
            gguf_close(gf);
            return -1;
        }
    }

    for (uint64_t i = 0; i < n_tensors; i++) {
        GGUFTensor *t = &gf->tensors[i];
        t->name = r_string(&c);
        if (!t->name) { gguf_close(gf); return -1; }
        if (!need(&c, 4)) { gguf_close(gf); return -1; }
        t->n_dims = (int)r_u32(&c);
        if (t->n_dims < 1 || t->n_dims > 4) {
            fprintf(stderr, "initium/gguf: tensor '%s' has invalid n_dims=%d\n", t->name, t->n_dims);
            gguf_close(gf);
            return -1;
        }
        for (int d = 0; d < t->n_dims; d++) {
            if (!need(&c, 8)) { gguf_close(gf); return -1; }
            t->ne[d] = r_u64(&c);
        }
        for (int d = t->n_dims; d < 4; d++) t->ne[d] = 1;
        if (!need(&c, 4 + 8)) { gguf_close(gf); return -1; }
        t->type = (int)r_u32(&c);
        t->offset = r_u64(&c);
        t->nbytes = ggml_nbytes(t->type, t->ne, t->n_dims);
    }

    /* align to 32 bytes for data section (GGUF alignment; default 32) */
    uint64_t alignment = 32;
    const GGUFKV *ak = gguf_find_kv(gf, "general.alignment");
    if (ak && ak->type == GGUF_TYPE_UINT32) alignment = ak->v.u32;

    size_t header_used = (size_t)(c.p - (const uint8_t *)gf->map);
    size_t pad = (alignment - (header_used % alignment)) % alignment;
    if (header_used + pad > gf->map_size) {
        fprintf(stderr, "initium/gguf: alignment past EOF\n");
        gguf_close(gf);
        return -1;
    }
    gf->data_base = (uint8_t *)gf->map + header_used + pad;

    for (int i = 0; i < gf->n_tensors; i++) {
        GGUFTensor *t = &gf->tensors[i];
        if (t->offset + t->nbytes > gf->map_size - (size_t)(gf->data_base - (uint8_t *)gf->map)
            && t->nbytes > 0) {
            /* soft check: offset is relative to data_base */
        }
        t->data = gf->data_base + t->offset;
        if ((uint8_t *)t->data + t->nbytes > (uint8_t *)gf->map + gf->map_size) {
            fprintf(stderr, "initium/gguf: tensor '%s' data out of bounds "
                    "(offset=%llu nbytes=%zu)\n",
                    t->name, (unsigned long long)t->offset, t->nbytes);
            gguf_close(gf);
            return -1;
        }
        (void)ggml_type_size;
    }

    return 0;
}

void gguf_close(GGUFFile *gf) {
    if (!gf) return;
    if (gf->kv) {
        for (int i = 0; i < gf->n_kv; i++) {
            free(gf->kv[i].key);
            if (gf->kv[i].type == GGUF_TYPE_STRING) free(gf->kv[i].v.str);
            if (gf->kv[i].type == GGUF_TYPE_ARRAY) {
                if (gf->kv[i].arr_type == GGUF_TYPE_STRING && gf->kv[i].v.arr) {
                    char **arr = (char **)gf->kv[i].v.arr;
                    for (uint64_t j = 0; j < gf->kv[i].arr_len; j++) free(arr[j]);
                }
                free(gf->kv[i].v.arr);
            }
        }
        free(gf->kv);
    }
    if (gf->tensors) {
        for (int i = 0; i < gf->n_tensors; i++) free(gf->tensors[i].name);
        free(gf->tensors);
    }
    unmap_file(gf);
    memset(gf, 0, sizeof(*gf));
}

const GGUFKV *gguf_find_kv(const GGUFFile *gf, const char *key) {
    for (int i = 0; i < gf->n_kv; i++) {
        if (strcmp(gf->kv[i].key, key) == 0) return &gf->kv[i];
    }
    return NULL;
}

const GGUFTensor *gguf_find_tensor(const GGUFFile *gf, const char *name) {
    for (int i = 0; i < gf->n_tensors; i++) {
        if (strcmp(gf->tensors[i].name, name) == 0) return &gf->tensors[i];
    }
    return NULL;
}

int gguf_get_u32(const GGUFFile *gf, const char *key, uint32_t *out) {
    const GGUFKV *k = gguf_find_kv(gf, key);
    if (!k) return -1;
    if (k->type == GGUF_TYPE_UINT32) { *out = k->v.u32; return 0; }
    if (k->type == GGUF_TYPE_INT32)  { *out = (uint32_t)k->v.i32; return 0; }
    if (k->type == GGUF_TYPE_UINT64) { *out = (uint32_t)k->v.u64; return 0; }
    return -1;
}

int gguf_get_i32(const GGUFFile *gf, const char *key, int32_t *out) {
    const GGUFKV *k = gguf_find_kv(gf, key);
    if (!k) return -1;
    if (k->type == GGUF_TYPE_INT32)  { *out = k->v.i32; return 0; }
    if (k->type == GGUF_TYPE_UINT32) { *out = (int32_t)k->v.u32; return 0; }
    return -1;
}

int gguf_get_f32(const GGUFFile *gf, const char *key, float *out) {
    const GGUFKV *k = gguf_find_kv(gf, key);
    if (!k) return -1;
    if (k->type == GGUF_TYPE_FLOAT32) { *out = k->v.f32; return 0; }
    if (k->type == GGUF_TYPE_FLOAT64) { *out = (float)k->v.f64; return 0; }
    return -1;
}

int gguf_get_str(const GGUFFile *gf, const char *key, const char **out) {
    const GGUFKV *k = gguf_find_kv(gf, key);
    if (!k || k->type != GGUF_TYPE_STRING) return -1;
    *out = k->v.str;
    return 0;
}

int gguf_get_u32_arr(const GGUFFile *gf, const char *key, const uint32_t **data, uint64_t *len) {
    const GGUFKV *k = gguf_find_kv(gf, key);
    if (!k || k->type != GGUF_TYPE_ARRAY) return -1;
    if (k->arr_type != GGUF_TYPE_UINT32 && k->arr_type != GGUF_TYPE_INT32) return -1;
    *data = (const uint32_t *)k->v.arr;
    *len = k->arr_len;
    return 0;
}
