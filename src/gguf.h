#ifndef INITIUM_GGUF_H
#define INITIUM_GGUF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF v3 subset used by Initium */

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} GGUFType;

/* ggml type ids we care about */
typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q6_K = 14,
} GGMLType;

typedef struct {
    char     *name;
    int       n_dims;
    uint64_t  ne[4];
    int       type;       /* GGMLType */
    uint64_t  offset;     /* relative to data section start */
    void     *data;       /* pointer into mmap */
    size_t    nbytes;
} GGUFTensor;

typedef struct {
    char   *key;
    int     type;         /* GGUFType (or array subtype in arr_type) */
    int     arr_type;
    uint64_t arr_len;
    /* value storage */
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        uint64_t u64;
        int64_t  i64;
        double   f64;
        int      b;
        char    *str;
        void    *arr;     /* raw array payload (typed) */
    } v;
} GGUFKV;

typedef struct {
    void     *map;
    size_t    map_size;
    int       version;
    int       n_kv;
    GGUFKV   *kv;
    int       n_tensors;
    GGUFTensor *tensors;
    uint8_t  *data_base;  /* start of tensor data (aligned) */
#ifdef _WIN32
    void     *win_file;
    void     *win_map;
#endif
} GGUFFile;

int  gguf_open(const char *path, GGUFFile *gf);
void gguf_close(GGUFFile *gf);

const GGUFKV    *gguf_find_kv(const GGUFFile *gf, const char *key);
const GGUFTensor *gguf_find_tensor(const GGUFFile *gf, const char *name);

/* Typed metadata getters — return 0 on success, -1 if missing */
int gguf_get_u32(const GGUFFile *gf, const char *key, uint32_t *out);
int gguf_get_i32(const GGUFFile *gf, const char *key, int32_t *out);
int gguf_get_f32(const GGUFFile *gf, const char *key, float *out);
int gguf_get_str(const GGUFFile *gf, const char *key, const char **out);
int gguf_get_u32_arr(const GGUFFile *gf, const char *key, const uint32_t **data, uint64_t *len);

#ifdef __cplusplus
}
#endif

#endif /* INITIUM_GGUF_H */
