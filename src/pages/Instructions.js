import { Layout, Typography, Menu, Card } from 'antd';
import { Link } from 'react-router-dom';

const { Header, Content, Footer } = Layout;
const { Title, Paragraph } = Typography;

function Instructions() {
  return (
    <Layout>
      {/* Navigation Bar */}
      <Header style={{ background: '#001529', display: 'flex', alignItems: 'center', padding: '0 20px' }}>
        <Title level={3} style={{ color: 'white', margin: '0', flex: 1 }}>KD-Pruning Simulator</Title>
        <Menu theme="dark" mode="horizontal" style={{ flex: 1, justifyContent: 'flex-end', display: 'flex', gap: '15px' }}>
          <Menu.Item key="1"><Link to="/">Home</Link></Menu.Item>
          <Menu.Item key="2"><Link to="/instructions">Instructions</Link></Menu.Item>
          <Menu.Item key="3"><Link to="/upload">Upload Dataset</Link></Menu.Item>
          <Menu.Item key="4"><Link to="/training">Training</Link></Menu.Item>
          <Menu.Item key="5"><Link to="/evaluation">Evaluation</Link></Menu.Item>
          <Menu.Item key="6"><Link to="/visualization">Visualization</Link></Menu.Item>
         
        </Menu>
      </Header>

      {/* Main Content */}
      <Content style={{ padding: '50px', textAlign: 'center', maxWidth: 900, margin: '0 auto' }}>
        <Title>How to Use the KD-Pruning Simulator</Title>
        <Paragraph>
          This simulator allows you to explore <b>Knowledge Distillation</b> and <b>Model Pruning</b> techniques interactively.
          Follow the steps below to start:
        </Paragraph>

        {/* Instructions Card */}
        <Card style={{ padding: 30, borderRadius: 12, boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.1)', textAlign: 'left' }}>
          <Paragraph>
            <b>1️⃣ Upload Your Dataset:</b> Go to the <Link to="/upload">Upload Dataset</Link> page and provide your dataset. <br /><br />
            <b>2️⃣ Train Your Model:</b> Navigate to the <Link to="/training">Training</Link> page to train a student model using KD. <br /><br />
            <b>3️⃣ Evaluate Performance:</b> Check accuracy and efficiency on the <Link to="/evaluation">Evaluation</Link> page. <br /><br />
            <b>4️⃣ Visualize Results:</b> Explore the impact of KD & Pruning on the <Link to="/visualization">Visualization</Link> page.
          </Paragraph>
        </Card>
      </Content>

      {/* Footer */}
      <Footer style={{ textAlign: 'center', background: '#001529', color: 'white', padding: '20px' }}>
        © 2025 KD-Pruning Simulator. All rights reserved.
      </Footer>
    </Layout>
  );
}

export default Instructions;
