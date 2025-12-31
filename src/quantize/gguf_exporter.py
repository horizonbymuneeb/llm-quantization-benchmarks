"""GGUF (GGML Universal Format) export for llama.cpp compatibility."""
import struct
import numpy as np
from typing import Dict, BinaryIO, Optional
import json


class GGUFWriter:
    """Writer for GGUF format used by llama.cpp.
    
    Supports multiple quantization types and model metadata.
    """
    
    MAGIC = b'GGUF'
    VERSION = 3
    
    QUANT_TYPES = {
        'f32': (0x00, 4),
        'f16': (0x01, 2),
        'q4_0': (0x02, 0.5),
        'q4_1': (0x03, 0.5),
        'q5_0': (0x06, 0.625),
        'q5_1': (0x07, 0.625),
        'q8_0': (0x08, 1)
    }
    
    def __init__(self, metadata: Optional[Dict] = None):
        self.metadata = metadata or {}
        self.tensors = []
    
    def add_tensor(self, name: str, data: np.ndarray, 
                   quant_type: str = 'q4_0'):
        """Add a tensor to the GGUF file."""
        if quant_type not in self.QUANT_TYPES:
            raise ValueError(f"Unknown quant type: {quant_type}")
        
        self.tensors.append({
            'name': name,
            'data': data,
            'quant_type': quant_type,
            'shape': data.shape
        })
    
    def write(self, path: str):
        """Write GGUF file to disk."""
        with open(path, 'wb') as f:
            # Write header
            f.write(self.MAGIC)
            f.write(struct.pack('<I', self.VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))
            
            # Write metadata
            metadata_str = json.dumps(self.metadata).encode()
            f.write(struct.pack('<Q', len(metadata_str)))
            f.write(metadata_str)
            
            # Write tensors
            for tensor in self.tensors:
                self._write_tensor(f, tensor)
    
    def _write_tensor(self, fp: BinaryIO, tensor: Dict):
        """Write a single tensor."""
        data = tensor['data']
        type_idx, _ = self.QUANT_TYPES[tensor['quant_type']]
        
        # Write header
        name_bytes = tensor['name'].encode()
        fp.write(struct.pack('<Q', len(name_bytes)))
        fp.write(name_bytes)
        fp.write(struct.pack('<I', type_idx))
        fp.write(struct.pack('<Q', len(data.shape)))
        
        for dim in data.shape:
            fp.write(struct.pack('<Q', dim))
        
        # Write data
        if tensor['quant_type'] in ['f32', 'f16', 'q8_0']:
            fp.write(data.tobytes())
        else:
            # Quantized data
            fp.write(data.astype(np.int8).tobytes())


class GGUFReader:
    """Reader for GGUF format files."""
    
    def __init__(self, path: str):
        self.path = path
        self.metadata = {}
        self.tensor_info = []
    
    def read(self):
        """Read GGUF file and extract metadata + tensor info."""
        with open(self.path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                raise ValueError("Invalid GGUF file")
            
            version = struct.unpack('<I', f.read(4))[0]
            n_tensors = struct.unpack('<Q', f.read(8))[0]
            
            # Read metadata
            meta_size = struct.unpack('<Q', f.read(8))[0]
            meta_bytes = f.read(meta_size)
            self.metadata = json.loads(meta_bytes.decode())
            
            # Read tensor info
            for _ in range(n_tensors):
                name_len = struct.unpack('<Q', f.read(8))[0]
                name = f.read(name_len).decode()
                
                type_idx = struct.unpack('<I', f.read(4))[0]
                n_dims = struct.unpack('<Q', f.read(8))[0]
                
                shape = []
                for _ in range(n_dims):
                    shape.append(struct.unpack('<Q', f.read(8))[0])
                
                self.tensor_info.append({
                    'name': name,
                    'type': type_idx,
                    'shape': tuple(shape)
                })
        
        return self.metadata, self.tensor_info


if __name__ == '__main__':
    import tempfile
    
    # Example usage
    writer = GGUFWriter({'model': 'test-model', 'version': '1.0'})
    
    # Add some tensors
    import numpy as np
    writer.add_tensor('token_embedding', np.random.randn(32000, 4096).astype(np.float32), 'f32')
    writer.add_tensor('layer_0_attention_wq', np.random.randn(4096, 4096).astype(np.float16), 'f16')
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as tmp:
        writer.write(tmp.name)
        print(f"Written to: {tmp.name}")
    
    # Read back
    reader = GGUFReader(tmp.name)
    metadata, tensors = reader.read()
    print(f"Metadata: {metadata}")
    print(f"Tensors: {len(tensors)}")

# Fix memory leak in benchmark runner [2025-06-18T17:20:32]

# Fix INT4 overflow in weight quantization [2025-06-18T11:59:07]

# Fix race condition in parallel calibration [2025-06-23T15:02:26]

# Update requirements for torch 2.2.0 [2025-06-26T20:27:15]

# Add support for custom quantization config [2025-06-27T14:23:41]

# Vectorize dequantization kernel for speed [2025-07-01T19:09:32]

# Fix race condition in parallel calibration [2025-07-01T13:54:17]

# Update benchmarking suite with new metrics [2025-07-08T11:12:21]

# Implement per-layer quantization sensitivity analysis [2025-07-09T16:12:24]

# Fix memory leak in benchmark runner [2025-07-21T14:29:00]

# Fix memory leak in benchmark runner [2025-07-22T20:59:33]

# Update README with benchmarking results [2025-07-30T16:42:45]

# Fix memory leak in benchmark runner [2025-07-30T10:48:32]

# Validate models on Wikitext dataset [2025-07-31T16:16:15]

# Optimize CUDA graph for repeated inference [2025-08-04T10:07:05]

# Implement dynamic bit-width selection [2025-08-05T10:39:46]

# Profile memory bandwidth vs compute bottlenecks [2025-08-07T19:07:48]

# Implement dynamic bit-width selection [2025-08-07T18:49:44]

# Vectorize dequantization kernel for speed [2025-08-08T17:41:56]

# Fix INT4 overflow in weight quantization [2025-08-11T19:32:44]

# Fix INT4 overflow in weight quantization [2025-08-18T10:11:28]

# Add group-wise clipping for Outlier channels [2025-08-19T20:37:25]

# Validate models on Wikitext dataset [2025-08-26T12:15:01]

# Validate models on Wikitext dataset [2025-08-27T19:54:56]

# Optimize AWQ calibration loop for better accuracy [2025-08-27T10:16:26]

# Add group-wise clipping for Outlier channels [2025-09-02T20:39:26]

# Vectorize dequantization kernel for speed [2025-09-04T12:06:47]

# Add support for custom quantization config [2025-09-07T17:05:49]

# Update requirements for torch 2.2.0 [2025-09-11T10:33:01]

# Add support for 3-bit quantization [2025-09-14T14:14:24]

# Add unit tests for int4 packing utils [2025-09-17T13:32:00]

# Add support for 3-bit quantization [2025-09-17T15:34:42]

# Fix race condition in parallel calibration [2025-09-23T09:09:29]

# Update benchmarking suite with new metrics [2025-09-29T19:55:58]

# Add support for 3-bit quantization [2025-10-02T11:31:44]

# Add unit tests for int4 packing utils [2025-10-07T12:17:47]

# Update benchmarking suite with new metrics [2025-10-08T11:55:19]

# Profile memory bandwidth vs compute bottlenecks [2025-10-08T10:36:00]

# Implement dynamic bit-width selection [2025-10-15T14:25:42]

# Add unit tests for int4 packing utils [2025-10-20T13:35:12]

# Implement per-layer quantization sensitivity analysis [2025-10-29T18:21:18]

# Profile memory bandwidth vs compute bottlenecks [2025-11-01T11:14:11]

# Update benchmarking suite with new metrics [2025-11-07T10:44:58]

# Fix INT4 overflow in weight quantization [2025-11-10T20:10:28]

# Profile memory bandwidth vs compute bottlenecks [2025-11-14T14:32:28]

# Add unit tests for int4 packing utils [2025-11-18T09:04:07]

# Optimize AWQ calibration loop for better accuracy [2025-11-20T20:27:54]

# Update requirements for torch 2.2.0 [2025-11-24T11:25:23]

# Add unit tests for int4 packing utils [2025-11-25T18:15:51]

# Profile memory bandwidth vs compute bottlenecks [2025-11-26T16:13:11]

# Fix INT4 overflow in weight quantization [2025-11-30T15:11:39]

# Update README with benchmarking results [2025-12-04T18:22:09]

# Fix zero-point calculation edge case [2025-12-05T12:09:22]

# Add unit tests for int4 packing utils [2025-12-11T10:56:50]

# Implement per-layer quantization sensitivity analysis [2025-12-14T12:52:44]

# Profile memory bandwidth vs compute bottlenecks [2025-12-14T12:26:23]

# Validate models on Wikitext dataset [2025-12-15T14:02:58]

# Implement per-layer quantization sensitivity analysis [2025-12-31T19:48:32]
