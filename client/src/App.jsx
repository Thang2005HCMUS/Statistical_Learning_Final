import React, { useState } from 'react';
import axios from 'axios';
import { Loader2, ChevronLeft, ChevronRight, Upload } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000/api/detect';

function App() {
  // State quản lý form
  const [modelName, setModelName] = useState('yolo10s');
  const [weightType, setWeightType] = useState('best');
  const [uploadMode, setUploadMode] = useState('single'); // 'single', 'batch', 'video'
  
  // State quản lý file và preview
  const [files, setFiles] = useState([]);
  const [inputPreviews, setInputPreviews] = useState([]);
  const [results, setResults] = useState([]);
  
  // State quản lý UI
  const [isLoading, setIsLoading] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);

  // Xử lý khi người dùng chọn file
  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    if (selectedFiles.length === 0) return;

    setFiles(selectedFiles);
    setResults([]);
    setCurrentIndex(0);

    // Tạo preview URL cho input
    const previews = selectedFiles.map(file => URL.createObjectURL(file));
    setInputPreviews(previews);
  };

  // Gửi request lên server
  const handleSubmit = async () => {
    if (files.length === 0) return alert('Vui lòng chọn file trước!');
    
    setIsLoading(true);
    const formData = new FormData();
    formData.append('model_name', modelName);
    formData.append('weight_type', weightType);

    try {
      if (uploadMode === 'single') {
        formData.append('file', files[0]);
        const res = await axios.post(`${API_BASE_URL}/single-image`, formData, {
          responseType: 'blob',
        });
        setResults([URL.createObjectURL(res.data)]);
      } 
      else if (uploadMode === 'batch') {
        files.forEach(file => formData.append('files', file));
        const res = await axios.post(`${API_BASE_URL}/batch-images`, formData);
        setResults(res.data.results); // Mảng base64
      } 
      else if (uploadMode === 'video') {
        formData.append('file', files[0]);
        const res = await axios.post(`${API_BASE_URL}/video`, formData, {
          responseType: 'blob',
        });
        setResults([URL.createObjectURL(res.data)]);
      }
    } catch (error) {
      console.error('Lỗi khi gọi API:', error);
      alert('Có lỗi xảy ra, vui lòng kiểm tra console!');
    } finally {
      setIsLoading(false);
    }
  };

  // Nút lùi/tiến cho chế độ nhiều ảnh
  const handlePrev = () => setCurrentIndex(prev => Math.max(0, prev - 1));
  const handleNext = () => setCurrentIndex(prev => Math.min(files.length - 1, prev + 1));

  return (
    <div className="min-h-screen bg-gray-100 p-8 font-sans">
      <div className="max-w-6xl mx-auto bg-white p-6 rounded-lg shadow-lg">
        <h1 className="text-2xl font-bold mb-6 text-center text-gray-800">Hệ Thống Nhận Diện Đối Tượng</h1>

        {/* Thanh Menu Control */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6 bg-gray-50 p-4 rounded border">
          <select 
            className="p-2 border rounded outline-none focus:ring-2 focus:ring-blue-500"
            value={modelName} 
            onChange={(e) => setModelName(e.target.value)}
          >
            <option value="yolo10s">YOLOv10s</option>
            <option value="rtdetr">RT-DETR</option>
            <option value="faster-rcnn">Faster R-CNN</option>
          </select>

          <select 
            className="p-2 border rounded outline-none focus:ring-2 focus:ring-blue-500"
            value={weightType} 
            onChange={(e) => setWeightType(e.target.value)}
          >
            <option value="best">Best Weights (best.pt/pth)</option>
            <option value="last">Last Weights (last.pt/pth)</option>
          </select>

          <select 
            className="p-2 border rounded outline-none focus:ring-2 focus:ring-blue-500"
            value={uploadMode} 
            onChange={(e) => {
              setUploadMode(e.target.value);
              setFiles([]); setInputPreviews([]); setResults([]);
            }}
          >
            <option value="single">1 Ảnh (Single)</option>
            <option value="batch">Nhiều Ảnh (Batch)</option>
            <option value="video">Video</option>
          </select>

          <div className="flex gap-2">
            <label className="flex-1 flex items-center justify-center bg-gray-200 hover:bg-gray-300 text-gray-700 p-2 rounded cursor-pointer transition">
              <Upload className="w-5 h-5 mr-2" />
              Chọn file
              <input 
                type="file" 
                className="hidden" 
                accept={uploadMode === 'video' ? "video/*" : "image/*"}
                multiple={uploadMode === 'batch'}
                onChange={handleFileChange}
              />
            </label>
            <button 
              className="flex-1 bg-blue-600 hover:bg-blue-700 text-white p-2 rounded font-semibold transition disabled:opacity-50"
              onClick={handleSubmit}
              disabled={isLoading || files.length === 0}
            >
              Chạy AI
            </button>
          </div>
        </div>

        {/* Nút điều hướng cho chế độ nhiều ảnh */}
        {uploadMode === 'batch' && inputPreviews.length > 0 && (
          <div className="flex justify-center items-center gap-4 mb-4">
            <button onClick={handlePrev} disabled={currentIndex === 0} className="p-2 bg-gray-200 rounded-full disabled:opacity-50 hover:bg-gray-300">
              <ChevronLeft className="w-6 h-6" />
            </button>
            <span className="font-semibold text-gray-700">Ảnh {currentIndex + 1} / {files.length}</span>
            <button onClick={handleNext} disabled={currentIndex === files.length - 1} className="p-2 bg-gray-200 rounded-full disabled:opacity-50 hover:bg-gray-300">
              <ChevronRight className="w-6 h-6" />
            </button>
          </div>
        )}

        {/* Vùng hiển thị chia 2: Input - Output */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-[500px]">
          {/* Input Panel */}
          <div className="border-2 border-dashed border-gray-300 rounded flex items-center justify-center bg-gray-50 overflow-hidden relative relative">
            <span className="absolute top-2 left-2 bg-black/50 text-white px-2 py-1 rounded text-sm z-10">Input</span>
            {inputPreviews.length > 0 ? (
              uploadMode === 'video' ? (
                <video src={inputPreviews[0]} controls className="max-w-full max-h-full object-contain" />
              ) : (
                <img src={inputPreviews[currentIndex]} alt="Input" className="max-w-full max-h-full object-contain" />
              )
            ) : (
              <span className="text-gray-400">Chưa có dữ liệu đầu vào</span>
            )}
          </div>

          {/* Output Panel */}
          <div className="border-2 border-gray-300 rounded flex items-center justify-center bg-gray-100 overflow-hidden relative">
            <span className="absolute top-2 left-2 bg-blue-600 text-white px-2 py-1 rounded text-sm z-10">Kết quả</span>
            
            {isLoading ? (
              <div className="flex flex-col items-center text-blue-600">
                <Loader2 className="w-12 h-12 animate-spin mb-2" />
                <span className="font-medium animate-pulse">Server đang xử lý...</span>
              </div>
            ) : results.length > 0 ? (
              uploadMode === 'video' ? (
                <video src={results[0]} controls autoPlay loop className="max-w-full max-h-full object-contain" />
              ) : (
                <img src={results[currentIndex]} alt="Output" className="max-w-full max-h-full object-contain" />
              )
            ) : (
              <span className="text-gray-400">Kết quả sẽ hiển thị ở đây</span>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;