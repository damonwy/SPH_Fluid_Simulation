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
	float computeWeight(const std::shared_ptr<Particle> &particle);
	float computeViscosity(const std::shared_ptr<Particle> &particle);
	float computePressure(const std::shared_ptr<Particle> &particle);
	Eigen::Vector3f computeGradWeight(const std::shared_ptr<Particle> &particle);

	// Static, shared by all particles
	static void init(int n);
	static void findNeighbors(const std::vector< std::shared_ptr<Particle> > &particles);
	static void centering(const std::vector< std::shared_ptr<Particle> > &particles);
	static void updateDensity(const std::vector< std::shared_ptr<Particle> > &particles);
	static void updateSPHForces(const std::vector< std::shared_ptr<Particle> > &particles);
	static void matchVelocity(const std::vector< std::shared_ptr<Particle> > &particles);
	static void seekGoal(const Eigen::Vector3f goal_pos, const std::vector< std::shared_ptr<Particle> > &particles);
	static void avoidCollision(const std::vector< std::shared_ptr<Particle> > &particles);
	static void draw(const std::vector< std::shared_ptr<Particle> > &particles,
					 std::shared_ptr<Program> prog);
	static float randFloat(float l, float h);
	static Eigen::Vector3f limitValue(Eigen::Vector3f vec, float maxVal);

private:
	// Properties that are fixed
	Eigen::Map<Eigen::Vector3f> color; // color (mapped to a location in colBuf)
	float &scale;                      // size (mapped to a location in scaBuf)
	int idx;
	
	// Properties that changes every rebirth
	float m;        // mass
	float h;		// radius
	
	float d;        // viscous damping
	float lifespan; // how long this particle lives
	float tEnd;     // time this particle dies
	int type;		// determine target or boid
	
	// Properties that changes every frame
	Eigen::Map<Eigen::Vector3f> x; // position (mapped to a location in posBuf)
	Eigen::Vector3f v;             // velocity
	float rho;				  // density
	Eigen::Vector3f A;				// acceleration
	float pres;				// pressure

	Eigen::Vector3f f;        // total force
	Eigen::Vector3f f_pv;	  // pressure force and viscosity
	Eigen::Vector3f f_a;	  // collision avoidance
	Eigen::Vector3f f_c;	  // centering
	Eigen::Vector3f f_m;      // velocity matching
	Eigen::Vector3f f_g;      // goal seeking
	float &alpha;             // mapped to a location in alpBuf
	float viscosity_coeff;
	float epsilon;			// 
	float rho_bar;      
	float upsilon;
	float pres_bar;
	int num_neighbors;		// number of nearby neighbors

	// Static, shared by all particles
	static std::vector<float> posBuf;
	static std::vector<float> colBuf;
	static std::vector<float> alpBuf;
	static std::vector<float> scaBuf;
	static std::vector<int> neighbourList; // a list of index of neighbors
	static GLuint posBufID;
	static GLuint colBufID;
	static GLuint alpBufID;
	static GLuint scaBufID;
};

#endif