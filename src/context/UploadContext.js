import { createContext, useState } from "react";

export const UploadContext = createContext();

export const UploadProvider = ({ children }) => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [evaluationResults, setEvaluationResults] = useState(null);

  return (
    <UploadContext.Provider value={{ uploadedFile, setUploadedFile, evaluationResults, setEvaluationResults }}>
      {children}
    </UploadContext.Provider>
  );
};
