#include <cassert>
#include <cstring>
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>

#include <omp.h>
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Camera.h"
#include "GLSL.h"
#include "MatrixStack.h"
#include "Particle.h"
#include "Program.h"
#include "Texture.h"
#include "sph.h"
#include "Node.h"
#include "Shape.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

GLFWwindow *window; // Main application window
string RESOURCE_DIR = "./"; // Where the resources are loaded from
Eigen::Vector3f window_llc, window_urc;
shared_ptr<Camera> camera;
shared_ptr<Program> prog;
shared_ptr<Program> progSimple;
shared_ptr<Program> prog_2;
shared_ptr<Texture> texture0;
shared_ptr<SPH> sph_demo;
vector< shared_ptr< Particle> > particles;
vector< shared_ptr< Node> > balls;
shared_ptr<Shape> sphereShape;

char pixels[4 * 1920 * 1080];

Eigen::Vector3f grav;
float t, h;

bool keyToggles[256] = {false}; // only for English keyboards!


// This function is called when a GLFW error occurs
static void error_callback(int error, const char *description)
{
	cerr << description << endl;
}

// This function is called when a key is pressed
static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

// This function is called when the mouse is clicked
static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
	// Get the current mouse position.
	double xmouse, ymouse;
	glfwGetCursorPos(window, &xmouse, &ymouse);
	// Get current window size.
	int width, height;
	glfwGetWindowSize(window, &width, &height);
	if(action == GLFW_PRESS) {
		bool shift = (mods & GLFW_MOD_SHIFT) != 0;
		bool ctrl  = (mods & GLFW_MOD_CONTROL) != 0;
		bool alt   = (mods & GLFW_MOD_ALT) != 0;
		camera->mouseClicked((float)xmouse, (float)ymouse, shift, ctrl, alt);
	}
}

// This function is called when the mouse moves
static void cursor_position_callback(GLFWwindow* window, double xmouse, double ymouse)
{
	int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	if(state == GLFW_PRESS) {
		camera->mouseMoved((float)xmouse, (float)ymouse);
	}
}

static void char_callback(GLFWwindow *window, unsigned int key)
{
	keyToggles[key] = !keyToggles[key];
}

// If the window is resized, capture the new size and reset the viewport
static void resize_callback(GLFWwindow *window, int width, int height)
{
	glViewport(0, 0, width, height);
}

// This function is called once to initialize the scene and OpenGL
static void init()
{
	// Initialize time.
	glfwSetTime(0.0);
	
	// Set background color.
	glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
	// Enable z-buffer test.
	glEnable(GL_DEPTH_TEST);
	// Enable setting gl_PointSize from vertex shader
	glEnable(GL_PROGRAM_POINT_SIZE);
	// Enable quad creation from sprite
	glEnable(GL_POINT_SPRITE);

	progSimple = make_shared<Program>();
	progSimple->setShaderNames(RESOURCE_DIR + "simple_vert.glsl", RESOURCE_DIR + "simple_frag.glsl");
	progSimple->setVerbose(true);
	progSimple->init();
	progSimple->addUniform("P");
	progSimple->addUniform("MV");
	
	prog = make_shared<Program>();
	prog->setShaderNames(RESOURCE_DIR + "vert.glsl", RESOURCE_DIR + "frag.glsl");
	//
	prog->setVerbose(true);
	prog->init();
	prog->addAttribute("aPos");
	prog->addAttribute("aAlp");
	prog->addAttribute("aCol");
	prog->addAttribute("aSca");

	prog->addUniform("P");
	prog->addUniform("MV");
	prog->addUniform("screenSize");
	prog->addUniform("texture0");

	prog_2 = make_shared<Program>();
	prog_2->setVerbose(true);
	prog_2->setShaderNames(RESOURCE_DIR + "phong_vert.glsl", RESOURCE_DIR + "phong_frag.glsl");
	prog_2->init();
	prog_2->addUniform("P");
	prog_2->addUniform("MV");
	prog_2->addUniform("kdFront");
	prog_2->addUniform("kdBack");
	prog_2->addAttribute("aPos");
	prog_2->addAttribute("aNor");
	prog_2->addAttribute("aTex");
	camera = make_shared<Camera>();
	camera->setInitDistance(10.0f);
	
	texture0 = make_shared<Texture>();
	texture0->setFilename(RESOURCE_DIR + "big.jpg");//alpha.jpg
	texture0->init();
	texture0->setUnit(0);
	texture0->setWrapModes(GL_REPEAT, GL_REPEAT);

	
	int num_particles = 20000;
	float init_x = 0.5f;
	float init_y = 3.5f;
	float h = 0.15f;
	float eps = 0.008f;
	float vis = 100.0f;
	float base_density = 15.9f;
	float power = 3.0f;
	float base_pressure = 5.0f;
	float dt = 0.2f;

	
	window_llc << 0.0f, 0.0f, 0.0f;
	window_urc << 4.0f, 4.0f, 4.0f;

	sph_demo = make_shared<SPH>(num_particles, init_x, init_y, h, eps, vis, base_density, base_pressure, power, dt, keyToggles);
	sph_demo->LLC = window_llc;
	sph_demo->URC = window_urc;
	sph_demo->init();

	// Init a sphere
	sphereShape = make_shared<Shape>();
	sphereShape->loadMesh(RESOURCE_DIR + "sphere2.obj");
	sphereShape->init();

	for (int i = 0; i < num_particles; i++) {
		auto sphere = make_shared<Node>(sphereShape);
		balls.push_back(sphere);
		sphere->r = 0.04;
		sphere->x = sph_demo->particles[i]->x;

	}
	


	t = 0.0f;
	

	GLSL::checkError(GET_FILE_LINE);
}

// This function is called every frame to draw the scene.
static void render()
{
	// Clear framebuffer.
	if (keyToggles[(unsigned)'z']) {
		glClear(GL_DEPTH_BUFFER_BIT);
	}
	else {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	if(keyToggles[(unsigned)'c']) {
		glEnable(GL_CULL_FACE);
	} else {
		glDisable(GL_CULL_FACE);
	}
	if(keyToggles[(unsigned)'l']) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	} else {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	
	// Get current frame buffer size.
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	camera->setAspect((float)width/(float)height);
	
	// Matrix stacks
	auto P = make_shared<MatrixStack>();
	auto MV = make_shared<MatrixStack>();
	
	// Apply camera transforms
	P->pushMatrix();
	camera->applyProjectionMatrix(P);
	MV->pushMatrix();
	camera->applyViewMatrix(MV);
	
	// Draw star of David
	progSimple->bind();
	glUniformMatrix4fv(progSimple->getUniform("P"), 1, GL_FALSE, glm::value_ptr(P->topMatrix()));
	glUniformMatrix4fv(progSimple->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
	glLineWidth(3.0f);
	/*glBegin(GL_LINE_LOOP);
	for (int i = 0; i < 6; ++i) {
		glColor3f(1.0f, 1.0f, 1.0f);
		glVertex3d(sin(i / 6.0 * 2 * 3.14159), 0.0f,
			cos(i / 6.0 * 2 * 3.14159));
	}
	glVertex3d(sin(6 / 6.0 * 2 * 3.14159), 0.0f,
		cos(6 / 6.0 * 2 * 3.14159));
	glVertex3d(sin(4 / 6.0 * 2 * 3.14159), 0.0f,
		cos(4 / 6.0 * 2 * 3.14159));
	glVertex3d(sin(2 / 6.0 * 2 * 3.14159), 0.0f,
		cos(2 / 6.0 * 2 * 3.14159));
	glVertex3d(sin(0 / 6.0 * 2 * 3.14159), 0.0f,
		cos(0/ 6.0 * 2 * 3.14159));

	glVertex3d(sin(1 / 6.0 * 2 * 3.14159), 0.0f,
		cos(1 / 6.0 * 2 * 3.14159));
	glVertex3d(sin(3 / 6.0 * 2 * 3.14159), 0.0f,
		cos(3 / 6.0 * 2 * 3.14159));
	glVertex3d(sin(5 / 6.0 * 2 * 3.14159), 0.0f,
		cos(5 / 6.0 * 2 * 3.14159));
	glVertex3d(sin(1 / 6.0 * 2 * 3.14159), 0.0f,
		cos(1 / 6.0 * 2 * 3.14159));
	glEnd();
*/

	// Draw Regular Box
	double boxxl = window_llc(0);
	double boxxh = window_urc(0);

	double boxyl = window_llc(1);
	double boxyh = window_urc(1);
	double boxzl = window_llc(2);
	double boxzh = window_urc(2);

	GLfloat box_diffuse[] = { 0.7, 0.7, 0.7 };
	GLfloat box_specular[] = { 0.1, 0.1, 0.1 };
	GLfloat box_shininess[] = { 1.0 };
	GLfloat ball_ambient[] = { 0.4, 0.0, 0.0 };
	GLfloat ball_diffuse[] = { 0.3, 0.0, 0.0 };
	GLfloat ball_specular[] = { 0.3, 0.3, 0.3 };
	GLfloat ball_shininess[] = { 10.0 };
	GLfloat box_ambient[] = { 0.1, 0.1, 0.1 };
	GLfloat smallr00[] = { 0.1, 0.0, 0.0 };
	GLfloat small0g0[] = { 0.0, 0.1, 0.0 };
	GLfloat small00b[] = { 0.0, 0.0, 0.1 };
	GLfloat smallrg0[] = { 0.1, 0.1, 0.0 };
	GLfloat smallr0b[] = { 0.1, 0.0, 0.1 };
	GLfloat small0gb[] = { 0.0, 0.1, 0.1 };
	GLfloat smallrgb[] = { 0.1, 0.1, 0.1 };
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//glPushMatrix();
	glEnable(GL_COLOR_MATERIAL);

	//Draw the box
	//set material parameters
	glMaterialfv(GL_FRONT, GL_AMBIENT, box_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, box_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, box_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, box_shininess);

	glBegin(GL_QUADS);
	//back face
	//
	//glColor3f(244/256, 158/256, 66/256);
	glColor3f(255.0 / 255.0f, 215.0 / 255.0f, 0.0f);
	glMaterialfv(GL_FRONT, GL_AMBIENT, smallrgb);
	glVertex3f(boxxl, boxyl, boxzl);
	glVertex3f(boxxh, boxyl, boxzl);
	glVertex3f(boxxh, boxyh, boxzl);
	glVertex3f(boxxl, boxyh, boxzl);

	//glColorMaterial(GL_FRONT, GL_DIFFUSE);

	//left face
	glColor3f(60.0/255.0f, 179.0/255.0f, 113.0/255.0f);

	glMaterialfv(GL_FRONT, GL_AMBIENT, small0g0);
	glVertex3f(boxxl, boxyl, boxzh);
	glVertex3f(boxxl, boxyl, boxzl);
	glVertex3f(boxxl, boxyh, boxzl);
	glVertex3f(boxxl, boxyh, boxzh);

	////glColorMaterial(GL_FRONT, GL_SPECULAR);
	//glColor3f(0.9, 0.1, 0.3);
	////right face
	//glMaterialfv(GL_FRONT, GL_AMBIENT, small00b);
	//glVertex3f(boxxh, boxyl, boxzh);
	//glVertex3f(boxxh, boxyh, boxzh);
	//glVertex3f(boxxh, boxyh, boxzl);
	//glVertex3f(boxxh, boxyl, boxzl);


	glColor3f(176.0/255.0f, 224.0/255.0f, 230.0/255.0f);
	//bottom face
	glMaterialfv(GL_FRONT, GL_AMBIENT, smallrg0);
	glVertex3f(boxxh, boxyl, boxzh);
	glVertex3f(boxxh, boxyl, boxzl);
	glVertex3f(boxxl, boxyl, boxzl);
	glVertex3f(boxxl, boxyl, boxzh);

	//top face
	/*glColor3f(0.8, 0.2, 0.5);
	glMaterialfv(GL_FRONT, GL_AMBIENT, smallr0b);
	glVertex3f(boxxh, boxyh, boxzh);
	glVertex3f(boxxl, boxyh, boxzh);
	glVertex3f(boxxl, boxyh, boxzl);
	glVertex3f(boxxh, boxyh, boxzl);*/

	//front face
	/*glColor3f(0.3, 0.4, 0.3);
	glMaterialfv(GL_FRONT, GL_AMBIENT, small0gb);
	glVertex3f(boxxh, boxyl, boxzh);
	glVertex3f(boxxl, boxyl, boxzh);
	glVertex3f(boxxl, boxyh, boxzh);
	glVertex3f(boxxh, boxyh, boxzh);*/

	glEnd();

	progSimple->unbind();

	// Draw particles
	glEnable(GL_BLEND);
	glDepthMask(GL_FALSE);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	prog->bind();
	texture0->bind(prog->getUniform("texture0"));
	glUniformMatrix4fv(prog->getUniform("P"), 1, GL_FALSE, glm::value_ptr(P->topMatrix()));
	glUniformMatrix4fv(prog->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
	glUniform2f(prog->getUniform("screenSize"), (float)width, (float)height);
	Particle::draw(sph_demo->particles, prog);
	texture0->unbind();
	prog->unbind();
	glDepthMask(GL_TRUE);
	glDisable(GL_BLEND);

	// Draw mesh
	prog_2->bind();
	glUniformMatrix4fv(prog_2->getUniform("P"), 1, GL_FALSE, glm::value_ptr(P->topMatrix()));
	MV->pushMatrix();
	sph_demo->draw(MV, prog_2);

	for (int i = 0; i < (int)balls.size(); ++i) {
		balls[i]->x = sph_demo->particles[i]->x;
		balls[i]->draw(MV, prog_2);
	}

	MV->popMatrix();
	prog_2->unbind();

	MV->popMatrix();
	P->popMatrix();

	GLSL::checkError(GET_FILE_LINE);
}

void stepParticles()
{	
	if (keyToggles[(unsigned)'d']) {
		sph_demo->den_bar *= 0.9;
		cout << "decrease base density to: " << sph_demo->den_bar << endl;
	}

	if (keyToggles[(unsigned)'D']) {
		sph_demo->den_bar *= 1.0 / 0.9;
		cout << "increase base density to: " << sph_demo->den_bar << endl;
	}

	if (keyToggles[(unsigned)'h']) {
		sph_demo->h *= 0.9;
		cout << "decrease h to: " << sph_demo->h << endl;
	}

	if (keyToggles[(unsigned)'H']) {
		sph_demo->h *= 1.0 / 0.9;
		cout << "increase h to: " << sph_demo->h << endl;
	}

	if (keyToggles[(unsigned)'w']) {
		sph_demo->wallsticky *= 0.9;
		cout << "decrease wallsticky to: " << sph_demo->wallsticky << endl;
	}

	if (keyToggles[(unsigned)'W']) {
		sph_demo->wallsticky *= 1.0 / 0.9;
		cout << "increase wallsticky to: " << sph_demo->wallsticky << endl;
	}
	if (keyToggles[(unsigned)'p']) {


		sph_demo->grav *= 0.9f;
		cout << "- grav" << grav.norm() << endl;
	}
	if (keyToggles[(unsigned)'P']) {
		
		
		sph_demo->grav *= 1.0/0.9f;
		cout << "+ grav" <<grav.norm() << endl;
	}
	if (keyToggles[(unsigned)'U']) {
		float mag = sph_demo->grav.norm();
		sph_demo->grav << 0.0f, 1.0f, 0.0;
		sph_demo->grav *= mag;
		cout << "change grav dir to up "  << endl;
	}

	if (keyToggles[(unsigned)'Z']) {
		float mag = sph_demo->grav.norm();
		sph_demo->grav << 0.0f, -1.0f, 0.0;
		sph_demo->grav *= mag;
		cout << "change grav dir to down " << endl;
	}

	if (keyToggles[(unsigned)'L']) {
		float mag = sph_demo->grav.norm();
		sph_demo->grav << -1.0f, 0.0f, 0.0;
		sph_demo->grav *= mag;
		cout << "change grav dir to left " << endl;
	}

	if (keyToggles[(unsigned)'R']) {
		float mag = sph_demo->grav.norm();
		sph_demo->grav << 1.0f, 0.0f, 0.0;
		sph_demo->grav *= mag;
		cout << "change grav dir to right " << endl;
	}

	if(keyToggles[(unsigned)' ']) {
		// This can be parallelized!
		sph_demo->updateFluid();
		std::string str = std::to_string(t);
		const char *cstr = str.c_str();
		glReadPixels(0, 0, (GLsizei)1920, (GLsizei)1080, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
		stbi_write_jpg(cstr, 1920, 1080, 4, pixels, 100);
		t += h;
	}
}

int main(int argc, char **argv)
{
	if(argc < 2) {
		cout << "Please specify the resource directory." << endl;
		return 0;
	}
	RESOURCE_DIR = argv[1] + string("/");

	// Set error callback.
	glfwSetErrorCallback(error_callback);
	// Initialize the library.
	if(!glfwInit()) {
		return -1;
	}
	// Create a windowed mode window and its OpenGL context.
	window = glfwCreateWindow(640, 480, "YING WANG", NULL, NULL);
	if(!window) {
		glfwTerminate();
		return -1;
	}
	// Make the window's context current.
	glfwMakeContextCurrent(window);
	// Initialize GLEW.
	glewExperimental = true;
	if(glewInit() != GLEW_OK) {
		cerr << "Failed to initialize GLEW" << endl;
		return -1;
	}
	glGetError(); // A bug in glewInit() causes an error that we can safely ignore.
	cout << "OpenGL version: " << glGetString(GL_VERSION) << endl;
	cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
	GLSL::checkVersion();
	// Set vsync.
	glfwSwapInterval(1);
	// Set keyboard callback.
	glfwSetKeyCallback(window, key_callback);
	// Set char callback.
	glfwSetCharCallback(window, char_callback);
	// Set cursor position callback.
	glfwSetCursorPosCallback(window, cursor_position_callback);
	// Set mouse button callback.
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	// Set the window resize call back.
	glfwSetFramebufferSizeCallback(window, resize_callback);
	// Initialize scene.
	init();
	// Loop until the user closes the window.
	while(!glfwWindowShouldClose(window)) {
		// Step particles.
		stepParticles();
		// Render scene.
		render();
		// Swap front and back buffers.
		glfwSwapBuffers(window);
		// Poll for and process events.
		glfwPollEvents();
	}
	// Quit program.
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}

