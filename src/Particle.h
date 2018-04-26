#pragma once
#ifndef _PARTICLE_H_
#define _PARTICLE_H_

#include <memory>
#include <vector>

#define GLEW_STATIC
#include <GL/glew.h>

#define EIGEN_DONT_ALIGN_STATICALLY
#include <Eigen/Dense>
class MatrixStack;
class Program;
class Texture;

class Particle
{
public:
	Particle(int index);
	virtual ~Particle();
	void rebirth(int type, float t, const bool *keyToggles);
	void step(float t, float h, const Eigen::Vector3f &g, const bool *keyToggles);
	Eigen::Vector3f v;				// velocity
	Eigen::Vector3f f;        // total force

	float m;        // mass
	float h;		// radius
	
	float d;        // viscous damping
	float lifespan; // how long this particle lives
	float tEnd;     // time this particle dies
	int type;		// determine target or boid
	float den;						// density
	float pres;						// pressure


	// Static, shared by all particles
	static void init(int n);
	static void draw(const std::vector< std::shared_ptr<Particle> > &particles, std::shared_ptr<Program> prog);
	static float randFloat(float l, float h);
	static Eigen::Vector3f limitValue(Eigen::Vector3f vec, float maxVal);
	Eigen::Map<Eigen::Vector3f> x; // position (mapped to a location in posBuf)
float &alpha;             // mapped to a location in alpBuf
float &scale;                      // size (mapped to a location in scaBuf)
private:
	// Properties that are fixed
	Eigen::Map<Eigen::Vector3f> color; // color (mapped to a location in colBuf)
	
	int idx;
	
	
	// Static, shared by all particles
	static std::vector<float> posBuf;
	static std::vector<float> colBuf;
	static std::vector<float> alpBuf;
	static std::vector<float> scaBuf;
	
	static GLuint posBufID;
	static GLuint colBufID;
	static GLuint alpBufID;
	static GLuint scaBufID;
};

#endif