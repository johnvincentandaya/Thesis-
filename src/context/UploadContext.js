import { createContext, useState } from "react";

export const UploadContext = createContext();

export const UploadProvider = ({ children }) => {
  const [uploadedFile, setUploadedFile] = useState(null);

  return (
    <UploadContext.Provider value={{ uploadedFile, setUploadedFile }}>
      {children}
    </UploadContext.Provider>
  );
};
