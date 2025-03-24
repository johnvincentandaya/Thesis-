import { Layout, Typography, Button, Menu, Tooltip, Card } from 'antd';
import { Link } from 'react-router-dom';

const { Header, Content, Footer } = Layout;
const { Title, Paragraph } = Typography;

function Home() {
  return (
    <Layout>
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
      <Content style={{ padding: '50px', textAlign: 'center' }}>
        <Title>Welcome to KD & Pruning Simulator</Title>
        <Paragraph>
          Explore how <Tooltip title="Knowledge Distillation transfers knowledge from a large teacher model to a smaller student model."><b>Knowledge Distillation</b></Tooltip>
          and <Tooltip title="Pruning removes less significant weights from the model to reduce size while maintaining accuracy."><b> Model Pruning</b></Tooltip> work through an interactive simulation.
        </Paragraph>
        <Card style={{ maxWidth: 800, margin: "20px auto", padding: 30, borderRadius: 10, boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.1)' }}>
          <Title level={3}>Understanding KD & Pruning</Title>
          <Paragraph>
            <b>Knowledge Distillation (KD)</b> is a technique where a large, complex model (teacher) transfers its learned knowledge to a smaller, more efficient model (student).
          </Paragraph>
          <Paragraph>
            <b>Model Pruning</b> removes less important connections from a neural network, reducing its size while maintaining most of its accuracy.
          </Paragraph>
        </Card>
        <Button type="primary" size="large" style={{ marginTop: '20px', transition: '0.3s', padding: '10px 20px' }}
          onMouseEnter={e => e.target.style.backgroundColor = '#1890ff'}
          onMouseLeave={e => e.target.style.backgroundColor = ''}>
          <Link to="/instructions" style={{ color: 'white' }}>Get Started</Link>
        </Button>
      </Content>
      <Footer style={{ textAlign: 'center', background: '#001529', color: 'white', padding: '20px' }}>
        © 2025 KD-Pruning Simulator. All rights reserved.
      </Footer>
    </Layout>
  );
}
export default Home;
