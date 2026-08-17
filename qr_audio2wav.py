#!/usr/bin/env python3
"""
EnCodec + QR Code 音频压缩工具
功能：将音频文件压缩为EnCodec格式，并拆分为多个二维码图片；或从二维码图片还原音频。
依赖安装：pip install encodec qrcode pillow opencv-python numpy librosa soundfile torchaudio reportlab pyzbar scipy
"""

import os
import sys
import tempfile
import glob
from pathlib import Path
from typing import List, Tuple
import qrcode
from PIL import Image
import numpy as np
import cv2
from encodec import EncodecModel
import torch
import librosa
from scipy.io import wavfile

# ================== 配置 ==================
QR_VERSION = 40  # 最大版本
QR_ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_L  # 最低纠错，容量最大
QR_CHUNK_SIZE = 2900  # 版本40 + 最低纠错 + 二进制模式的安全容量
SAMPLE_RATE = 24000
BANDWIDTH = 1.5  # 单位是 kbps，最低值 1.5

# ================== 核心压缩/解压 ==================

def compress_audio(input_path: str, output_path: str) -> None:
    print(f"📁 加载音频: {input_path}")
    
    # 加载模型
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(BANDWIDTH)
    print(f"🔧 目标码率: {BANDWIDTH} kbps")
    
    # 使用 librosa 读取音频，直接重采样到 24kHz
    wav_np, sr = librosa.load(input_path, sr=model.sample_rate, mono=True, dtype=np.float32)
    print(f"📊 采样率: {sr} Hz, 时长: {len(wav_np)/sr:.2f} 秒")
    
    # 转为 torch tensor，形状: (channels, samples)
    wav = torch.from_numpy(wav_np).unsqueeze(0)  # (1, samples)
    
    # 添加 batch 维度: (batch, channels, samples)
    wav = wav.unsqueeze(0)  # (1, 1, samples)
    
    print(f"📊 输入形状: {wav.shape}")
    
    # 编码
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    
    # 提取 codes - 直接从 encoded_frames 中提取整数索引
    codes_list = []
    for layer in encoded_frames:
        if isinstance(layer, tuple) and len(layer) > 0:
            codes = layer[0]  # 形状: (num_quantizers, frames)
            if isinstance(codes, torch.Tensor):
                codes_int = codes.to(torch.int16).cpu().numpy()
                codes_list.append(codes_int)
    
    if not codes_list:
        print("❌ 没有提取到 codes")
        return
    
    # 将所有 codes 合并
    if len(codes_list) > 1:
        all_codes = np.concatenate(codes_list, axis=0)
    else:
        all_codes = codes_list[0]
    
    # 去掉 batch 维度（如果有）
    if len(all_codes.shape) == 3:
        all_codes = all_codes.squeeze(0)
    
    print(f"📊 codes 形状: {all_codes.shape}")
    
    # 保存为二进制
    with open(output_path, 'wb') as f:
        num_quantizers, frames = all_codes.shape
        f.write(num_quantizers.to_bytes(2, 'little'))
        f.write(frames.to_bytes(4, 'little'))
        f.write(all_codes.tobytes())
    
    size = os.path.getsize(output_path) / 1024
    print(f"✅ 压缩完成: {output_path} ({size:.1f} KB)")


def decompress_audio(input_path: str, output_path: str) -> None:
    print(f"📂 解压音频: {input_path}")
    
    # 读取压缩数据
    with open(input_path, 'rb') as f:
        num_quantizers = int.from_bytes(f.read(2), 'little')
        frames = int.from_bytes(f.read(4), 'little')
        print(f"📊 num_quantizers: {num_quantizers}, frames: {frames}")
        
        data_bytes = f.read()
        all_codes = np.frombuffer(data_bytes, dtype=np.int16).reshape(num_quantizers, frames)
    
    # 转为 torch.long
    codes_tensor = torch.tensor(all_codes, dtype=torch.long)
    codes_tensor = codes_tensor.unsqueeze(0)
    
    print(f"📊 重组形状: {codes_tensor.shape}, dtype: {codes_tensor.dtype}")
    print(f"📊 值范围: {codes_tensor.min().item()} ~ {codes_tensor.max().item()}")
    
    # 加载模型
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(BANDWIDTH)
    
    # 解码
    with torch.no_grad():
        decoded = model.decode([(codes_tensor, None)])
    
    # 保存为 WAV（使用 scipy，最稳定）
    decoded_np = decoded.squeeze(0).cpu().numpy()  # 形状: (channels, samples)
    # 转置为 (samples, channels)
    if decoded_np.ndim == 2 and decoded_np.shape[0] == 1:
        decoded_np = decoded_np.squeeze(0)  # 单声道 -> (samples,)
    elif decoded_np.ndim == 2:
        decoded_np = decoded_np.T  # (samples, channels)
    
    # 转为 int16（WAV 标准格式）
    decoded_int16 = (decoded_np * 32767).astype(np.int16)
    wavfile.write(output_path, model.sample_rate, decoded_int16)
    print(f"✅ 解压完成: {output_path}")


# ================== QR码拆分与还原 ==================

def split_to_qr(data: bytes, prefix: str = "qr") -> List[str]:
    total_size = len(data)
    chunk_size = QR_CHUNK_SIZE
    
    print(f"📦 使用分块大小: {chunk_size} 字节")
    
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    total = len(chunks)
    
    print(f"🔢 数据分为 {total} 个二维码")
    
    if total > 40:
        print(f"❌ 数据太大（{total_size/1024:.1f}KB），需要进一步压缩")
        print(f"💡 建议：用更短的音频（如 30 秒）")
        sys.exit(1)
    
    paths = []
    for i, chunk in enumerate(chunks, 1):
        header = f"{i:04d}".encode()
        payload = header + chunk
        
        qr = qrcode.QRCode(
            version=QR_VERSION,
            error_correction=QR_ERROR_CORRECTION,
            box_size=5,
            border=2,
        )
        qr.add_data(payload)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        path = f"{prefix}_{i:02d}.png"
        img.save(path)
        paths.append(path)
        
        print(f"  ✅ 生成 {path} (序号 {i}/{total})")
    
    return paths


def restore_from_qr(qr_paths: List[str], output_file: str) -> bytes:
    chunks = []
    qr_paths = sorted(qr_paths)
    
    use_pyzbar = True
    try:
        from pyzbar.pyzbar import decode
        print("📖 使用 pyzbar 解码器")
    except ImportError:
        use_pyzbar = False
        print("⚠️ pyzbar 未安装，使用 OpenCV 解码（建议安装: pip install pyzbar）")
    
    for path in qr_paths:
        print(f"📖 扫描: {path}")
        
        if use_pyzbar:
            from pyzbar.pyzbar import decode
            img = Image.open(path)
            decoded = decode(img)
            if decoded:
                data = decoded[0].data.decode('utf-8')
            else:
                img_cv = cv2.imread(path)
                if img_cv is not None:
                    detector = cv2.QRCodeDetector()
                    data, bbox, _ = detector.detectAndDecode(img_cv)
                else:
                    data = ""
        else:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"无法读取图片: {path}")
            detector = cv2.QRCodeDetector()
            data, bbox, _ = detector.detectAndDecode(img)
        
        if not data:
            raise ValueError(f"QR码解码失败: {path}")
        
        if len(data) < 5:
            raise ValueError(f"数据太短: {path}")
        
        seq = int(data[:4])
        payload = data[4:].encode('latin-1')
        chunks.append((seq, payload))
        print(f"  ✅ 序号 {seq}, 数据 {len(payload)} 字节")
    
    chunks.sort(key=lambda x: x[0])
    full_data = b''.join(payload for _, payload in chunks)
    
    with open(output_file, 'wb') as f:
        f.write(full_data)
    
    print(f"✅ 还原完成: {output_file} ({len(full_data)/1024:.1f} KB)")
    return full_data


# ================== 工具函数 ==================

def layout_qr_codes(qr_paths: List[str], output_pdf: str = "qr_layout.pdf", 
                    per_page: int = 8, cols: int = 2):
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
    except ImportError:
        print("⚠️ 未安装reportlab, 跳过PDF排版")
        return
    
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4
    
    rows = (per_page + cols - 1) // cols
    margin = 15 * mm
    cell_w = (width - 2 * margin) / cols
    cell_h = (height - 2 * margin) / rows
    
    for idx, path in enumerate(qr_paths):
        if idx % per_page == 0 and idx > 0:
            c.showPage()
        
        pos = idx % per_page
        row = pos // cols
        col = pos % cols
        x = margin + col * cell_w + 5 * mm
        y = height - margin - (row + 1) * cell_h + 5 * mm
        size = min(cell_w, cell_h) - 10 * mm
        
        img = Image.open(path)
        temp_img = f"_temp_{idx}.png"
        img.save(temp_img)
        c.drawImage(temp_img, x, y, size, size)
        os.remove(temp_img)
        
        label = os.path.basename(path).replace('.png', '')
        c.setFont("Helvetica", 10)
        c.drawString(x, y - 5 * mm, f"#{label}")
    
    c.save()
    print(f"📄 PDF已生成: {output_pdf} ({len(qr_paths)} 个QR码)")


# ================== 主流程 ==================

def main():
    print("=" * 60)
    print("🎵 EnCodec + QR Code 音频工具")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\n用法:")
        print("  压缩: python qr_audio.py compress <音频文件> [输出前缀]")
        print("  解压: python qr_audio.py decompress <压缩文件> [输出WAV]")
        print("  还原: python qr_audio.py restore <qr图片列表> [输出压缩文件]")
        print("\n示例:")
        print("  python qr_audio.py compress song.mp3")
        print("  python qr_audio.py decompress song.ecdc")
        print("  python qr_audio.py restore qr_*.png")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "compress":
        input_file = sys.argv[2]
        prefix = sys.argv[3] if len(sys.argv) > 3 else "qr"
        
        with tempfile.NamedTemporaryFile(suffix=".ecdc", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            compress_audio(input_file, tmp_path)
            
            with open(tmp_path, 'rb') as f:
                data = f.read()
            
            qr_paths = split_to_qr(data, prefix)
            print(f"\n✅ 共生成 {len(qr_paths)} 个二维码图片:")
            for p in qr_paths:
                print(f"  - {p}")
            
            if len(qr_paths) <= 8:
                layout_qr_codes(qr_paths, f"{prefix}_layout.pdf")
            else:
                print(f"⚠️ QR码数量 ({len(qr_paths)}) 超过8个")
                print(f"💡 建议: 用更短的音频（如 30 秒），码数量可降到 2-3 个")
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    elif mode == "decompress":
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else "restored.wav"
        decompress_audio(input_file, output_file)
    
    elif mode == "restore":
        qr_pattern = sys.argv[2] if len(sys.argv) > 2 else "qr_*.png"
        qr_paths = glob.glob(qr_pattern)
        if not qr_paths:
            print(f"❌ 未找到匹配的QR码: {qr_pattern}")
            sys.exit(1)
        
        output_file = sys.argv[3] if len(sys.argv) > 3 else "restored.ecdc"
        restore_from_qr(qr_paths, output_file)
        
        auto_decompress = input("\n是否立即解压为音频? (y/n): ").lower()
        if auto_decompress == 'y':
            wav_path = output_file.replace('.ecdc', '.wav')
            decompress_audio(output_file, wav_path)
    
    else:
        print(f"❌ 未知模式: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
