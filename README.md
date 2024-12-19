
# 基于 InSwapper 换脸与 CodeFormer 图像提升的项目

  

本项目基于 **InSwapper** 换脸模型和 **CodeFormer** 图像提升模型，目标是提供一个完全支持 CPU 的版本，结合换脸和图像增强功能，同时修复了部分 InSwapper 原始代码中的 bug，并解决了 ONNX 模型调用的问题。

  

## 功能

  

- **换脸功能**：利用 InSwapper 模型进行高质量的面部交换。

- **图像提升**：使用 CodeFormer 模型提升图像质量和分辨率。

- **CPU 支持**：所有操作已转为仅支持 CPU。


  

## 项目结构

  

- `src/`：包含修改后的源代码。

  - **inswapper/**：核心的 InSwapper 换脸代码文件夹，已修改为完全支持 CPU 运行，并修复了与 ONNX 相关的问题。

  - **codeformer/**：CodeFormer 图像提升代码文件夹。

  - **utils/**：用于图像处理、模型加载和执行的辅助函数和工具脚本。

## 安装
### 先决条件

1. **Python 3.10+**

2. **必需的库**：

   - 通过 `requirements.txt` 文件安装。

### 安装步骤

1. 克隆仓库：
```bash
   git clone https://github.com/20070951/founder_faceswap.git

   cd founder_faceswap
```
2. 安装依赖：
```bash
   pip install -r requirements.txt
```

3. codeFormer安装：

    需要根据[Codeformer](https://github.com/sczhou/CodeFormer)的官方指南进行安装，主要是安装basicsr，这个库不要用pip直接安装，会出现错误；
## 修改内容

### InSwapper 代码修复：

- 修复了与 ONNX 运行时和执行相关的多个问题。
- 确保所有计算都在 CPU 上进行。

## 使用方法

- **换脸**：要在两张图片之间进行换脸，可以直接运行`src/` 目录中的 `face_swaper.py`该代码接受两张输入图片，生成换脸后的输出结果。
  
- **图像提升**：使用 `src/` 目录中的 `face_restoration.py` ，可以提高图像的质量和分辨率。

## 注意事项

- 本项目已将所有模型和计算操作转为仅支持 CPU，适用于没有 GPU 的环境。
- 若需要加速推理，建议安装支持 GPU 的环境并调整 ONNX 模型的提供者。
