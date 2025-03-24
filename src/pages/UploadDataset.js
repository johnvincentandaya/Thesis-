import React, { useState, useContext, } from "react";
import { Layout, Button, Card, Upload, Typography, message, Tooltip } from "antd";
import { UploadOutlined, ArrowLeftOutlined, InfoCircleOutlined } from "@ant-design/icons";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { UploadContext } from "../context/UploadContext";

const { Title, Paragraph, Text } = Typography;
const { Header, Content, Footer } = Layout;
const allowedFormats = [".csv", ".json", ".txt", ".xlsx", ".jpg", ".png", ".zip"];


const UploadDataset = () => {
    const navigate = useNavigate();
    const { setUploadedFile } = useContext(UploadContext);
    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [uploadSuccess, setUploadSuccess] = useState(false);

    const beforeUpload = (file) => {
        const fileExt = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    
        if (!allowedFormats.includes(fileExt)) {
            message.error("🚨 Invalid file type! Allowed formats: CSV, JSON, TXT, XLSX, JPG, PNG, ZIP.");
            return false;
        }
    
        setFile(file);
        return false; // Prevent automatic upload
    };

    
    const handleUpload = async () => {
        if (!file) {
            message.error("⚠️ Please select a file before uploading.");
            return;
        }
    
        setUploading(true);
        const formData = new FormData();
        formData.append("file", file);
    
        // Debugging logs
        console.log("Uploading file:", file);
        console.log("FormData contains:", formData.get("file"));
    
        try {
            const response = await axios.post("http://localhost:5000/upload", formData, {
                headers: { 
                    "Content-Type": "multipart/form-data" 
                },
                onUploadProgress: (progressEvent) => {
                    console.log(`Upload Progress: ${Math.round((progressEvent.loaded / progressEvent.total) * 100)}%`);
                }
            });
    
            console.log("Server Response:", response.data);
    
            if (response.data.success) {
                setUploadedFile(file);
                setUploadSuccess(true);
                message.success("✅ File uploaded successfully!");
            } else {
                message.error(response.data.message || "❌ Upload failed. Try again.");
            }
        } catch (error) {
            console.error("Upload Error:", error);
            message.error("🚨 Failed to upload file. Please check your connection.");
        } finally {
            setUploading(false);
        }
    };
    
    return (
        <Layout>
            {/* Navigation Bar */}
            <Header style={{ background: "#001529", display: "flex", alignItems: "center", padding: "0 20px" }}>
                <Title level={3} style={{ color: "white", margin: "0", flex: 1 }}>KD-Pruning Simulator</Title>
                <Button type="text" icon={<ArrowLeftOutlined />} onClick={() => navigate(-1)} style={{ color: "white" }}>
                    Back
                </Button>
            </Header>

            {/* Main Content */}
            <Content style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "80vh", padding: "20px" }}>
                <Card
                    title="📂 Upload Dataset"
                    bordered={false}
                    style={{ maxWidth: 500, width: "100%", textAlign: "center", borderRadius: 12, boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)" }}
                >
                    <Paragraph>
                        Select a CSV file to upload and use for training.
                    </Paragraph>

                    <Tooltip title="Ensure the file format is .csv before uploading.">
                        <InfoCircleOutlined style={{ fontSize: 18, color: "#1890ff", marginBottom: 10 }} />
                    </Tooltip>

                    <Upload beforeUpload={beforeUpload} showUploadList={false} accept={allowedFormats.join(",")}>
    <Button type="dashed" icon={<UploadOutlined />} style={{ width: "100%", marginBottom: 10 }}>
        {file ? `Selected: ${file.name}` : "Choose File"}
    </Button>
</Upload>

                    <Button
                        type="primary"
                        icon={<UploadOutlined />}
                        loading={uploading}
                        onClick={handleUpload}
                        disabled={!file}
                        style={{ width: "100%", transition: "0.3s" }}
                    >
                        {uploading ? "Uploading..." : "Upload"}
                    </Button>
                    <Button
                     type="primary"
                     onClick={() => navigate("/visualization")}
                     style={{ width: "100%", marginTop: 10 }}
                     disabled={!uploadSuccess} // ✅ Button is disabled until uploadSuccess is true
                    >
                    Next Step ➡️
                    </Button>
                    
                </Card>
            </Content>

            {/* Footer */}
            <Footer style={{ textAlign: "center", background: "#001529", color: "white", padding: "20px" }}>
                © 2025 KD-Pruning Simulator. All rights reserved.
            </Footer>
        </Layout>
    );
};

export default UploadDataset;
