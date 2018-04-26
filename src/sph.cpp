#include "sph.h"
#include "Particle.h"

#include <iostream>
#include <cmath>

#include <omp.h>
using namespace std;
using namespace Eigen;
const float  PI_F = 3.14159265358979f;

SPH::SPH(int n,
	float l,
	float u,
	float h,
	float _epsilon,
	float _viscosity_coeff,
	float _den_bar,
	float _pres_bar,
	float _upsilon,
	float _dt,
	const bool *keyToggles) {

	this->update_scheme = EL;
	this->epsilon = _epsilon;
	this->viscosity_coeff = _viscosity_coeff;
	this->den_bar = _den_bar;
	this->upsilon = _upsilon;
	this->pres_bar = _pres_bar;
	this->llc << l, l, l;
	this->urc << u, u, u;
	this->h = h;
	this->dt = _dt;

	this->grav << 0.0f, -10.0f, 0.0f;
	this->wallsticky = 0.5f;

	Vector3f v;
	v.setZero();

	Vector3f x;
	x.setZero();

	// init particles
	Particle::init(n);

	for (int i = 0; i < n; ++i) {
		auto p = make_shared<Particle>(i);
		p->x << randFloat(l + 1.0f, u - 1.0f), randFloat(l + 1.0f, u - 1.0f), randFloat(l + 1.0f, u - 1.0f);
		p->h = h;
		particles.push_back(p);p->rebirth(1, 0.0f, keyToggles);
	
		
	}

	updateOVboundary(llc, urc);
	updateOVindices();
}

float SPH::getWeight(shared_ptr<Particle> pa, shared_ptr<Particle> pb) {
	float w;
	float r = (pa->x - pb->x).norm();
	float h = pa->h;

	if (r / h < 1.0) {
		w = pow((1.0 - r / h), 3) * 10.0f / (pow(h, 2) * PI_F);
	}
	else {
		w = 0.0f;
	}
	return w;
}

Vector3f SPH::getGradWeight(shared_ptr<Particle> pa, shared_ptr<Particle> pb) {
	Vector3f gw;
	gw.setZero();

	float r = (pa->x - pb->x).norm();
	float h = pa->h;

	if (r / h < 1.0f) {
		gw = -pow((1.0f - r / h), 2)* 30.0f / (PI_F * pow(h, 3)) * (pa->x - pb->x).normalized();
	}
	return gw;
}

void SPH::updateDensity() {
	for (int i = 0; i < (int)particles.size(); i++) {

		vector<size_t> neighbor_indices;
		auto pa = particles[i];

		// neighbor_indices has all the indices of nearby particles
		findNeighbors(pa, &neighbor_indices);
		pa->den = 0.0f;

		for (int j = 0; j < neighbor_indices.size(); j++) {
			int ib = neighbor_indices[j];
			auto pb = particles[ib];
			pa->den += pb->m * getWeight(pa, pb);
		}
		//cout << pa->den << endl;
		// compute pressure using Tait equation where
		// pres_bar: strength  upsilon: power  den_bar: base density
		pa->pres = pres_bar * (pow(pa->den / den_bar, upsilon) - 1.0f);
	}
}

void SPH::updateOVboundary(Vector3f _llc, Vector3f _urc) {

	nx = int(((_urc - _llc)(0) / h) + 1);
	ny = int(((_urc - _llc)(1) / h) + 1);
	nz = int(((_urc - _llc)(2) / h) + 1);

	dx = (_urc - _llc)(0) / (float)(nx - 1);
	dy = (_urc - _llc)(1) / (float)(ny - 1);
	dz = (_urc - _llc)(2) / (float)(nz - 1);
	occupancy_volume.clear();
	occupancy_volume.resize(nx * ny * nz);
}

void SPH::updateOVindices() {

	for (int i = 0; i < (int)particles.size(); i++) {
		auto pa = particles[i];
		int ix, iy, iz;
		int id = getGridIndex(pa, ix, iy, iz);
		occupancy_volume[id].push_back(i);
	}
}

float SPH::computeSV(shared_ptr<Particle> pa, shared_ptr<Particle> pb) {
	Vector3f rab = pa->x - pb->x;
	Vector3f vab = pa->v - pb->v;

	float vr = rab.dot(vab);
	float SV;
	float gamma = viscosity_coeff * pa->h / (pa->den + pb->den);
	if (vr < 0.0f) {
		SV = -gamma * vr / (pow(rab.norm(), 2) + epsilon * pow(pa->h, 2));
	}
	else {
		SV = 0.0f;
	}
	return SV;
}

float SPH::computeSP(shared_ptr<Particle> pa, shared_ptr<Particle> pb) {
	float SP;
	SP = pa->pres / pow(pa->den, 2) + pb->pres / pow(pb->den, 2);
	return SP;
}

void SPH::updateForces() {

	for (int i = 0; i < (int)particles.size(); i++) {
		Vector3f force;
		force.setZero();

		Vector3f gw;	// gradient weight
		float SP, SV; 	// parameters for force_pressure, force_viscosity

		vector<size_t> neighbor_indices;
		auto pa = particles[i];
		findNeighbors(pa, &neighbor_indices);

		// add forces from neighbors
		for (int j = 0; j < neighbor_indices.size(); j++) {

			int ib = neighbor_indices[j];
			auto pb = particles[ib];
			gw = getGradWeight(pa, pb);
			SP = computeSP(pa, pb);
			SV = computeSV(pa, pb);
			force -= pb->m * (SP + SV) * gw;
		}
		force *= 1e-3;
		//cout << "f:" << force.norm() << endl;
		// add gravity
		force += grav;
		pa->f = force;
	}
}

void SPH::findNeighbors(shared_ptr<Particle> pa, vector<size_t> *neighbor_indices) {

	// For particle i want a list of particle indices for particles with 2h
	int grid_id;
	int ix, iy, iz;
	grid_id = getGridIndex(pa, ix, iy, iz);

	int l, r, u, d, in, out;
	l = ix - 1;
	r = ix + 1;
	u = iy + 1;
	d = iy - 1;
	in = iz - 1;
	out = iz + 1;

	vector<size_t> nearby_grids;

	int index;

	//nearby_grids[i] stores the indices of grids around grid i
	/*
	-------------
	|	|
	-------------
	|i  |
	-------------
	|	|
	-------------
	*/
if (r < nx && u <  ny && in >= 0) {
		index = r + nx * u + nx * ny * in;
		nearby_grids.push_back((size_t)index);
	}

	if (l >= 0 && u < ny && out < nz) {
		index = l + nx * u + nx * ny * out;
		nearby_grids.push_back((size_t)index);
	}


	// first check the eight corners
	if (l >= 0 && u < ny && in >= 0) {
		index = l + nx * u + nx * ny * in;
		nearby_grids.push_back((size_t)index);
	}

	if (l >= 0 && d >= 0 && in >= 0) {
		index = l + nx * d + nx * ny * in;
		nearby_grids.push_back((size_t)index);
	}

	if (r < nx && d >= 0 && in >= 0) {
		index = r + nx * d + nx * ny * in;
		nearby_grids.push_back((size_t)index);
	}

	

	if (l >= 0 && d >= 0 && out < nz) {
		index = l + nx * d + nx * ny * out;
		nearby_grids.push_back((size_t)index);
	}

	if (r < nx && d >= 0 && out < nz) {
		index = r + nx * d + nx * ny * out;
		nearby_grids.push_back((size_t)index);
	}

	if (r < nx && u <  ny && out < nz) {
		index = r + nx * u + nx * ny * out;
		nearby_grids.push_back((size_t)index);
	}
// then check four 12 grids
	if (r < nx && out < nz) {
		index = r + nx * iy + nx * ny * out;
		nearby_grids.push_back((size_t)index);
	}

	if (r < nx && u <  ny ) {
		index = r + nx * u + nx * ny * iz;
		nearby_grids.push_back((size_t)index);
	}

	if (r < nx && in >= 0) {
		index = r + nx * iy + nx * ny * in;
		nearby_grids.push_back((size_t)index);
	}

	if (r < nx && d >= 0) {
		index = r + nx * d + nx * ny * iz;
		nearby_grids.push_back((size_t)index);
	}

	if (u < ny && in >= 0) {
		index = ix + nx * u + nx * ny * in;
		nearby_grids.push_back((size_t)index);
	}

	if (u < ny && out < nz) {
		index = ix + nx *u + nx * ny * out;
		nearby_grids.push_back((size_t)index);
	}

	if (d >= 0 && in >= 0) {
		index = ix + nx * d + nx * ny * in;
		nearby_grids.push_back((size_t)index);
	}

	if (d >= 0 && out < nz) {
		index = ix + nx * d + nx * ny * out;
		nearby_grids.push_back((size_t)index);
	}

	if (l >= 0 && out < nz) {
		index = l + nx * iy + nx * ny * out;
		nearby_grids.push_back((size_t)index);
	}

	if (l >= 0 && in >= 0) {
		index = l + nx * iy + nx * ny * in;
		nearby_grids.push_back((size_t)index);
	}

	if (l >= 0 && d >= 0) {
		index = l + nx * d + nx * ny * iz;
		nearby_grids.push_back((size_t)index);
	}

	if (l >= 0 &&u < ny) {
		index = l + nx * u + nx * ny * iz;
		nearby_grids.push_back((size_t)index);
	}

	if (l >= 0) {
		index = l + nx * iy + nx * ny * iz;
		nearby_grids.push_back((size_t)index);
	}

	if (r<nx) {
		index = r + nx * iy + nx * ny * iz;
		nearby_grids.push_back((size_t)index);
	}

	if (u < ny) {
		index = ix + nx * u + nx * ny * iz;
		nearby_grids.push_back((size_t)index);
	}

	if (d >= 0) {
		index =ix + nx * d + nx * ny * iz;
		nearby_grids.push_back((size_t)index);
	}

	if (in >= 0) {
		index = ix + nx * iy + nx * ny * in;
		nearby_grids.push_back((size_t)index);
	}

	if (out<nz) {
		index = ix + nx * iy + nx * ny * out;
		nearby_grids.push_back((size_t)index);
	}

	nearby_grids.push_back((size_t)grid_id);

	for (int i = 0; i < nearby_grids.size(); i++) {

		int n_pts = occupancy_volume[nearby_grids[i]].size();

		for (int j = 0; j < n_pts; j++) {

			neighbor_indices->push_back(occupancy_volume[nearby_grids[i]][j]);
		}
	}
}

void SPH::checkBoundary(float wallsticky) {

	for (int i = 0; i < (int)particles.size(); i++) {
		auto pa = particles[i];
		if (pa->x(1) > URC(1) / 2.0) {
			pa->alpha = 0.3f;
			pa->scale = 0.1f;
		}

		// modify the velocity and position of particles when out of boundary
		if (pa->x(0) < LLC(0)) {
			pa->x(0) = LLC(0) + (LLC(0) - pa->x(0)) * wallsticky;
			pa->v(0) = -pa->v(0) * wallsticky;
		}
		if (pa->x(0) > URC(0)) {
			pa->x(0) = URC(0) + (URC(0) - pa->x(0)) * wallsticky;
			pa->v(0) = -pa->v(0) * wallsticky;
		}
		if (pa->x(1) < LLC(1)) {
			pa->x(1) = LLC(1) + (LLC(1) - pa->x(1)) * wallsticky;
			pa->v(1) = -pa->v(1) * wallsticky;
		}
		if (pa->x(1) > URC(1)) {
			pa->x(1) = URC(1) + (URC(1) - pa->x(1)) * wallsticky;
			pa->v(1) = -pa->v(1) * wallsticky;
		}
		if (pa->x(2) < LLC(2)) {
			pa->x(2) = LLC(2) + (LLC(2) - pa->x(2)) * wallsticky;
			pa->v(2) = -pa->v(2) * wallsticky;
		}
		if (pa->x(2) > URC(2)) {
			pa->x(2) = URC(2) + (URC(2) - pa->x(2)) * wallsticky;
			pa->v(2) = -pa->v(2) * wallsticky;
		}
		// update the occupancy volume boundary
		if (pa->x(0) < llc(0)) {
			llc(0) = pa->x(0);
		}
		if (pa->x(0) > urc(0)) {
			urc(0) = pa->x(0);
		}
		if (pa->x(1) < llc(1)) {
			llc(1) = pa->x(1);
		}
		if (pa->x(1) > urc(1)) {
			urc(1) = pa->x(1);
		}
		if (pa->x(2) < llc(2)) {
			llc(2) = pa->x(2);
		}
		if (pa->x(2) > urc(2)) {
			urc(2) = pa->x(2);
		}

	}
}

int SPH::getGridIndex(shared_ptr<Particle> pa, int &ix, int &iy, int &iz) {

	// compute the index of a particle given an occupancy volume grid
	Vector3f pos = pa->x;

	ix = int((pos - llc)(0) / (dx));
	iy = int((pos - llc)(1) / (dy));
	iz = int((pos - llc)(2) / (dz));
	int index = ix + nx * iy + nx * ny * iz;
	return index;
}

void SPH::updateFluid() {

	if (update_scheme == LF) {
		leapfrog(dt);
	}

	if (update_scheme == SIXTH) {
		sixth(dt);
	}

	if (update_scheme == EL) {
		euler(dt);
	}
}

void SPH::euler(float dt) {

	updateDensity();
	//cout << "den:" << particles[100]->den << endl;
	updateForces();
#pragma omp parallel for
	for (int i = 0; i < (int)particles.size(); i++) {
		auto pa = particles[i];
		pa->v += pa->f * dt;
		//cout << "v" << pa->v << endl;
		pa->x += pa->v * dt;
	}

	checkBoundary(wallsticky);
	updateOVboundary(llc, urc);
	updateOVindices();
}

void SPH::leapfrog(float _dt) {
	// non-implicit method

	updateDensity();
	updateForces();

	_dt = _dt / 2.0f;
#pragma omp parallel for
	for (int i = 0; i < (int)particles.size(); i++) {
		auto pa = particles[i];
		pa->x += pa->v * _dt; // half step
	}

	checkBoundary(wallsticky);
	updateOVboundary(llc, urc);
	updateOVindices();

	updateDensity();
	updateForces();
#pragma omp parallel for
	for (int i = 0; i < (int)particles.size(); i++) {
		auto pa = particles[i];
		pa->v += pa->f * _dt * 2.0f; // full step
		pa->x += pa->v * _dt;		// half step
	}

	checkBoundary(wallsticky);
	updateOVboundary(llc, urc);
	updateOVindices();
}

void SPH::sixth(float dt) {
	float a, b;
	a = 1.0f / (4.0f - pow(4.0f, 1.0f / 3.0f));
	b = 1.0f - 4.0f * a;

	for (int i = 0; i < 5; i++) {
		leapfrog(a * dt);
	}
}

float SPH::randFloat(float l, float h) {

	float r = rand() / (float)RAND_MAX;
	return (1.0f - r) * l + r * h;
}

SPH::~SPH() {

}