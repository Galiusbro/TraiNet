import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import TrainPage from "./pages/TrainPage";
import StatusPage from "./pages/StatusPage";
import WorkersPage from "./pages/WorkersPage";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <nav className="bg-white shadow-lg">
          <div className="max-w-7xl mx-auto px-4">
            <div className="flex justify-between h-16">
              <div className="flex">
                <div className="flex-shrink-0 flex items-center">
                  <span className="text-xl font-bold">TrainNet.ai</span>
                </div>
                <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                  <Link
                    to="/"
                    className="text-gray-900 inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-gray-300"
                  >
                    Обучить модель
                  </Link>
                  <Link
                    to="/workers"
                    className="text-gray-900 inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-gray-300"
                  >
                    Воркеры
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </nav>

        <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <Routes>
            <Route path="/" element={<TrainPage />} />
            <Route path="/status/:taskId" element={<StatusPage />} />
            <Route path="/workers" element={<WorkersPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
