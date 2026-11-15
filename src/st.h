/* st.h — safetensors 文件读取器（只读，mmap）。
 *
 * 文件布局：
 *   [0, 8)        小端 uint64，JSON 头的字节数 N
 *   [8, 8+N)      JSON 头：{"张量名": {"dtype":..., "shape":[...], "data_offsets":[begin,end]}, ...}
 *   [8+N, EOF)    数据区；data_offsets 相对数据区起点
 */
#ifndef ST_H
#define ST_H

#include <stddef.h>
#include <stdint.h>

#define ST_MAX_DIMS 8

typedef enum {
    ST_BOOL, ST_U8, ST_I8, ST_I16, ST_U16, ST_F16, ST_BF16,
    ST_I32, ST_U32, ST_F32, ST_I64, ST_U64, ST_F64,
    ST_DTYPE_COUNT
} st_dtype;

typedef struct {
    char    *name;                 /* UTF-8，以 0 结尾，归 st_file 所有 */
    st_dtype dtype;
    int      ndim;
    uint64_t shape[ST_MAX_DIMS];
    const uint8_t *data;           /* 指向 mmap 区域内部，只读 */
    uint64_t nbytes;
} st_tensor;

typedef struct {
    st_tensor *tensors;            /* 按名字的字节序升序排列 */
    size_t     count;
    void      *map;                /* 整个文件的映射 */
    size_t     map_size;
} st_file;

/* 成功返回 0。失败返回 -1，并把原因写进 err（可为 NULL）。 */
int  st_open(st_file *f, const char *path, char *err, size_t errcap);
void st_close(st_file *f);

const st_tensor *st_find(const st_file *f, const char *name);

/* 一个模型的权重通常拆成多个文件（分片）。st_set 把它们合并成一张名字表。 */
typedef struct {
    st_file          *files;
    size_t            nfiles;
    const st_tensor **tensors;     /* 指向各 files[i].tensors 内部，按名字升序 */
    size_t            count;
} st_set;

/* 打开 n 个分片。任何一个打不开，或同一个名字出现在两个分片里，都算失败。 */
int  st_set_open(st_set *s, const char *const *paths, size_t n, char *err, size_t errcap);
void st_set_close(st_set *s);
const st_tensor *st_set_find(const st_set *s, const char *name);
const char *st_dtype_name(st_dtype t);
size_t      st_dtype_size(st_dtype t);

#endif
