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
