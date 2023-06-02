import React, { useState, useEffect } from "react";
import axios from "axios";
import { Button, Form, Container, Navbar, Nav, Table, NavDropdown } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './FileUpload.scss';

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [data, setData] = useState([]);
  const [backgroundFade, setBackgroundFade] = useState(true); // State for background fade presence

  const submitFile = (event) => {
    event.preventDefault();

    const formData = new FormData();
    formData.append("file", file);

    axios
      .post("/your-api-endpoint", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((response) => {
        setUploadStatus("File uploaded successfully");
      })
      .catch((error) => {
        setUploadStatus("File upload failed");
      });
  };

  const handleFileUpload = (event) => {
    setFile(event.target.files[0]);
  };

  const fetchData = () => {
    // Fetch data from your API endpoint and update the 'data' state
    // Example:
    axios
      .get("/your-data-api-endpoint")
      .then((response) => {
        setData(response.data);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
      });
  };

  // Fetch initial data on component mount
  useEffect(() => {
    fetchData();
  }, []);

  // Toggle handler for enabling/disabling background fade
  const handleBackgroundFadeToggle = () => {
    setBackgroundFade(!backgroundFade);
  };

  // Update the className of the body element
  useEffect(() => {
    if (backgroundFade) {
      document.body.classList.add("background-fade");
    } else {
      document.body.classList.remove("background-fade");
    }
  }, [backgroundFade]);

  return (
    <>
      <Navbar bg="light" expand="lg">
      <Container fluid>
        <Navbar.Brand href="#">Neural Net Nexus</Navbar.Brand>
        <Navbar.Toggle aria-controls="navbarScroll" />
        <Navbar.Collapse id="navbarScroll">
          <Nav
            className="me-auto my-2 my-lg-0"
            style={{ maxHeight: '100px' }}
            navbarScroll
          >
            <Nav.Link href="#action1">Home</Nav.Link>
            <Nav.Link href="#action2">Link</Nav.Link>
            <NavDropdown title="Link" id="navbarScrollingDropdown">
              <NavDropdown.Item href="#action3">Action</NavDropdown.Item>
              <NavDropdown.Item href="#action4">
                Another action
              </NavDropdown.Item>
              <NavDropdown.Divider />
              <NavDropdown.Item href="#action5">
                Something else here
              </NavDropdown.Item>
            </NavDropdown>
            <Nav.Link href="#" disabled>
              Link
            </Nav.Link>
          </Nav>
          <Form className="d-flex">
            <Form.Control
              type="search"
              placeholder="Search"
              className="me-2"
              aria-label="Search"
            />
            <Button variant="outline-success">Search</Button>
          </Form>
        </Navbar.Collapse>
      </Container>
    </Navbar>
      <Container className="mt-5">
        <div className="upload-container">
          <h2>Upload .zip File</h2>
          <Form onSubmit={submitFile}>
            <Form.Group controlId="formFile" className="mb-3">
              <Form.Label>Select a .zip file</Form.Label>
              <Form.Control type="file" accept=".zip" onChange={handleFileUpload} />
            </Form.Group>
            <Button variant="primary" type="submit">Upload</Button>
            {uploadStatus && <p>{uploadStatus}</p>}
          </Form>
        </div>
        <Table striped bordered hover className={'mt-5'}> {/* Add fade-enabled class */}
          <thead>
            <tr>
              <th>#</th>
              <th>Column 1</th>
              <th>Column 2</th>
              <th>Column 3</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item, index) => (
              <tr key={index}>
                <td>{index + 1}</td>
                <td>{item.column1}</td>
                <td>{item.column2}</td>
                <td>{item.column3}</td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Container>
      <div className="text-center fixed-bottom pb-3 ">
          <label className="toggle-label"><h5>Toggle Background Fade</h5> </label>
          <input type="checkbox" checked={backgroundFade} onChange={handleBackgroundFadeToggle} />
        </div>
    </>
  );
};

export default FileUpload;
