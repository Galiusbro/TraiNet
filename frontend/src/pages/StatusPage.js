import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";

const STATUS_COLORS = {
  pending: "bg-yellow-100 text-yellow-800",
  queued: "bg-blue-100 text-blue-800",
  running: "bg-indigo-100 text-indigo-800",
  done: "bg-green-100 text-green-800",
  error: "bg-red-100 text-red-800",
};

function StatusPage() {
  const { taskId } = useParams();
  const [task, setTask] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await axios.get(
          `http://localhost:8000/api/tasks/${taskId}/status`
        );
        setTask(response.data);
      } catch (err) {
        setError(err.message);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000); // Обновляем каждые 5 секунд

    return () => clearInterval(interval);
  }, [taskId]);

  if (error) {
    return (
      <div className="bg-red-50 p-4 rounded-md">
        <div className="text-red-700">Ошибка: {error}</div>
      </div>
    );
  }

  if (!task) {
    return (
      <div className="text-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div>
        <p className="mt-4 text-gray-600">Загрузка...</p>
      </div>
    );
  }

  return (
    <div className="bg-white shadow sm:rounded-lg">
      <div className="px-4 py-5 sm:p-6">
        <h3 className="text-lg font-medium leading-6 text-gray-900">
          Статус задачи #{task.id}
        </h3>

        <div className="mt-5">
          <dl className="grid grid-cols-1 gap-5 sm:grid-cols-2">
            <div className="px-4 py-5 bg-gray-50 shadow rounded-lg overflow-hidden sm:p-6">
              <dt className="text-sm font-medium text-gray-500 truncate">
                Модель
              </dt>
              <dd className="mt-1 text-3xl font-semibold text-gray-900">
                {task.model_type}
              </dd>
            </div>

            <div className="px-4 py-5 bg-gray-50 shadow rounded-lg overflow-hidden sm:p-6">
              <dt className="text-sm font-medium text-gray-500 truncate">
                Статус
              </dt>
              <dd className="mt-1">
                <span
                  className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                    STATUS_COLORS[task.status]
                  }`}
                >
                  {task.status}
                </span>
              </dd>
            </div>
          </dl>
        </div>

        {task.result_model_url && (
          <div className="mt-5">
            <a
              href={task.result_model_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Скачать модель (Supabase URL)
            </a>
          </div>
        )}

        {task.status === "done" && task.result_model_filename && (
          <div className="mt-5">
            <a
              href={`http://localhost:8000/api/tasks/${task.id}/download_model`}
              download
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Скачать модель
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

export default StatusPage;
