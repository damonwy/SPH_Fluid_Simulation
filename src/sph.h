#pragma once
#include <memory>
#include <vector>

#define GLEW_STATIC
#include <GL/glew.h>

#define EIGEN_DONT_ALIGN_STATICALLY
#include <Eigen/Dense>


class Particle;
enum Scheme { EL, LF, SIXTH };

class SPH {

public:
	SPH(int n, float l, float u, float h, float _epsilon, float _viscosity_coeff, float _den_bar, float _pres_bar, float _upsilon, float _dt, const bool *keyToggles);
	~SPH();

	// fixed values
	Eigen::Vector3f LLC;
	Eigen::Vector3f URC;

	// computed each time step
	Eigen::Vector3f llc;
	Eigen::Vector3f urc;
	int nx;
	int ny;
	int nz;
	float dx;
	float dy;
	float dz;

	// user input
	float epsilon;
	float viscosity_coeff;
	float upsilon;
	float den_bar;
	float pres_bar;
	float h;
	float dt;
	float wallsticky;
	Eigen::Vector3f grav;
	Scheme update_scheme;
	std::vector<std::shared_ptr<Particle> > particles;
	std::vector<std::vector<size_t> > occupancy_volume; // list of nearby particles = occupancy_volume[i]

	void updateFluid();

private:
	void sixth(float dt);
	void leapfrog(float _dt);
	void euler(float dt);

	// apply to all particles
	void updateDensity();
	void updateForces();
	void checkBoundary(float wallsticky);
	void updateOVindices();

	// apply to one particle
	void findNeighbors(std::shared_ptr<Particle> pa, std::vector<size_t> *neighbor_indices);
	int getGridIndex(std::shared_ptr<Particle> pa, int &ix, int &iy, int &iz);

	// apply between two particles
	float computeSV(std::shared_ptr<Particle> pa, std::shared_ptr<Particle> pb);
	float computeSP(std::shared_ptr<Particle> pa, std::shared_ptr<Particle> pb);
	float getWeight(std::shared_ptr<Particle> pa, std::shared_ptr<Particle> pb);
	Eigen::Vector3f getGradWeight(std::shared_ptr<Particle> pa, std::shared_ptr<Particle> pb);

	// helper
	float randFloat(float l, float h);
	void updateOVboundary(Eigen::Vector3f _llc, Eigen::Vector3f _urc);
};