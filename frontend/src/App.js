import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import FileUpload from './FileUpload';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<FileUpload />} />
      </Routes>
    </Router>
  );
}

export default App;
