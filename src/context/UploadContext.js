import { createContext, useState } from "react";

export const UploadContext = createContext();

export const UploadProvider = ({ children }) => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [isUploaded, setIsUploaded] = useState(false);

  return (
    <UploadContext.Provider value={{ 
      uploadedFile, 
      setUploadedFile, 
      evaluationResults, 
      setEvaluationResults,
      isUploaded,
      setIsUploaded 
    }}>
      {children}
    </UploadContext.Provider>
  );
};
