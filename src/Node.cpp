#include <iostream>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Node.h"
#include "Shape.h"
#include "Program.h"
#include "MatrixStack.h"

using namespace std;



Node::Node() :
	r(1.0f),
	m(1.0f),
	i(-1),
	x(0.0f, 0.0f, 0.0f),
	v(0.0f, 0.0f, 0.0f),
	normal(0.0f, 0.0f, 0.0f),
	fixed(true),
	color(0.0f, 1.0f, 0.0f),
	scale(0.2f),
	alpha(0.5f)
{

}

Node::Node(const shared_ptr<Shape> s) :
	r(1.0f),
	m(1.0f),
	i(-1),
	x(0.0f, 0.0f, 0.0f),
	v(0.0f, 0.0f, 0.0f),
	normal(0.0f, 0.0f, 0.0f),
	fixed(true),
	sphere(s),
	color(0.0f, 1.0f, 0.0f),
	scale(0.2f),
	alpha(0.5f)
{

}

Node::~Node()
{
}

void Node::tare()
{
	x0 = x;
	v0 = v;
}

void Node::reset()
{
	x = x0;
	v = v0;
}

void Node::draw(shared_ptr<MatrixStack> MV, const shared_ptr<Program> prog) const
{
	if (sphere) {
		MV->pushMatrix();
		MV->translate(x(0), x(1), x(2));
		MV->scale(r);
		glUniformMatrix4fv(prog->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
		sphere->draw(prog);
		MV->popMatrix();
	}
}
