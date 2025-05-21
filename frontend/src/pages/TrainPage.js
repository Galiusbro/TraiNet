import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { supabase } from "../lib/supabase";

const MODEL_TYPES = [
  { id: "mobilenet_v2", name: "MobileNetV2 (для изображений)" },
  { id: "distilbert", name: "DistilBERT (для текста)" },
];

function TrainPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [modelType, setModelType] = useState("mobilenet_v2");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const { getRootProps, getInputProps } = useDropzone({
    accept: {
      "text/csv": [".csv"],
      "application/json": [".json"],
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      setFile(acceptedFiles[0]);
    },
  });

  const uploadToSupabase = async (file) => {
    const fileExt = file.name.split(".").pop();
    const fileName = `${Math.random()}.${fileExt}`;
    const filePath = `datasets/${fileName}`;

    const { data, error } = await supabase.storage
      .from("trainnet")
      .upload(filePath, file, {
        cacheControl: "3600",
        upsert: false,
      });

    if (error) throw error;

    const {
      data: { publicUrl },
    } = supabase.storage.from("trainnet").getPublicUrl(filePath);

    return publicUrl;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Пожалуйста, загрузите датасет");
      return;
    }

    setIsLoading(true);
    setError(null);
    setUploadProgress(0);

    try {
      // Загрузка файла в Supabase Storage
      const datasetUrl = await uploadToSupabase(file);
      setUploadProgress(100);

      // Создание задачи обучения
      const response = await axios.post("http://localhost:8000/api/train", {
        model_type: modelType,
        dataset_url: datasetUrl,
        hyperparams: {
          epochs: 2,
          batch_size: 32,
        },
      });

      navigate(`/status/${response.data.id}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white shadow sm:rounded-lg">
      <div className="px-4 py-5 sm:p-6">
        <h3 className="text-lg font-medium leading-6 text-gray-900">
          Обучить новую модель
        </h3>

        <form onSubmit={handleSubmit} className="mt-5 space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Выберите тип модели
            </label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            >
              {MODEL_TYPES.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Загрузите датасет
            </label>
            <div
              {...getRootProps()}
              className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md"
            >
              <div className="space-y-1 text-center">
                <input {...getInputProps()} />
                <div className="flex text-sm text-gray-600">
                  <p className="pl-1">
                    {file
                      ? file.name
                      : "Перетащите файл сюда или нажмите для выбора"}
                  </p>
                </div>
                <p className="text-xs text-gray-500">CSV или JSON до 10MB</p>
              </div>
            </div>
          </div>

          {uploadProgress > 0 && (
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className="bg-indigo-600 h-2.5 rounded-full"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
          )}

          {error && <div className="text-red-600 text-sm">{error}</div>}

          <button
            type="submit"
            disabled={isLoading}
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
          >
            {isLoading ? "Запуск..." : "Обучить модель"}
          </button>
        </form>
      </div>
    </div>
  );
}

export default TrainPage;
