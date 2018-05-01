#define TETLIBRARY

#include "sph.h"
#include "Particle.h"
#include "Node.h"
#include "MatrixStack.h"
#include "Program.h"
#include "GLSL.h"
#include <iostream>
#include <cmath>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

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

	this->grav << 0.0f, -15.0f, 0.0f;
	this->wallsticky = 0.5f;

	Vector3f v;
	v.setZero();

	Vector3f x;
	x.setZero();

	// init particles
	Particle::init(n);

	for (int i = 0; i < n; ++i) {
		auto p = make_shared<Particle>(i);
		p->x << randFloat(l, u ), randFloat(l, u ), randFloat(l, u);
		p->h = h;
		particles.push_back(p);p->rebirth(1, 0.0f, keyToggles);
		
	}

	updateOVboundary(llc, urc);
	updateOVindices();

	in.load_ply("bunny100");
	tetrahedralize("pqz", &in, &out);
	nVerts = out.numberofpoints;
	nTriFaces = out.numberoftrifaces;
	nEdges = out.numberofedges;
	Vector3f tf;
	tf << 1.0f, 0.0f, 1.0f;
	 
	for (int i = 0; i < nVerts; i++) {
		auto p = make_shared<Node>();
		nodes.push_back(p);
		p->x << float(out.pointlist[3 * i]+tf(0)), float(out.pointlist[3 * i + 1]+tf(1)), float(out.pointlist[3 * i + 2]+tf(2));
		p->m = 0.1f;
	}

	// Build vertex buffers
	posBuf.clear();
	norBuf.clear();
	texBuf.clear();
	eleBuf.clear();

	posBuf.resize(nTriFaces * 9);
	norBuf.resize(nTriFaces * 9);
	eleBuf.resize(nTriFaces * 3);

	updatePosNor();
	for (int i = 0; i < nTriFaces; i++) {
		eleBuf[3 * i + 0] = 3 * i + 0;
		eleBuf[3 * i + 1] = 3 * i + 1;
		eleBuf[3 * i + 2] = 3 * i + 2;
	}

}

void SPH::init() {
	glGenBuffers(1, &posBufID);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size() * sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);

	glGenBuffers(1, &norBufID);
	glBindBuffer(GL_ARRAY_BUFFER, norBufID);
	glBufferData(GL_ARRAY_BUFFER, norBuf.size() * sizeof(float), &norBuf[0], GL_DYNAMIC_DRAW);

	glGenBuffers(1, &eleBufID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, eleBuf.size() * sizeof(unsigned int), &eleBuf[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	assert(glGetError() == GL_NO_ERROR);
}


void SPH::updatePosNor() {
	for (int iface = 0; iface < nTriFaces; iface++) {
		Vector3f p1 = nodes[out.trifacelist[3 * iface + 0]]->x;
		Vector3f p2 = nodes[out.trifacelist[3 * iface + 1]]->x;
		Vector3f p3 = nodes[out.trifacelist[3 * iface + 2]]->x;

		//Position
		Vector3f e1 = p2 - p1;
		Vector3f e2 = p3 - p1;
		Vector3f normal = e1.cross(e2);
		normal.normalize();

		for (int idx = 0; idx < 3; idx++) {
			posBuf[9 * iface + 0 + idx] = p1(idx);
			posBuf[9 * iface + 3 + idx] = p2(idx);
			posBuf[9 * iface + 6 + idx] = p3(idx);
			norBuf[9 * iface + 0 + idx] = normal(idx);
			norBuf[9 * iface + 3 + idx] = normal(idx);
			norBuf[9 * iface + 6 + idx] = normal(idx);
		}
	}

	// Update AABB Bounding Box
	computeAABB();

}

void SPH::draw(shared_ptr<MatrixStack> MV, const shared_ptr<Program> p)const {

	glUniform3fv(p->getUniform("kdFront"), 1, Vector3f(250.0/255.0f, 128.0/255.0f, 114.0/255.0f).data());
	//glUniform3fv(p->getUniform("kdBack"), 1, Vector3f(1.0, 1.0, 0.0).data());
	glUniform3fv(p->getUniform("kdBack"), 1, Vector3f(255.0 / 255.0f, 182.0 / 255.0f, 193.0 / 255.0f).data());
	MV->pushMatrix();
	glUniformMatrix4fv(p->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
	int h_pos = p->getAttribute("aPos");
	glEnableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);

	glBufferData(GL_ARRAY_BUFFER, posBuf.size() * sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(h_pos, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);

	int h_nor = p->getAttribute("aNor");
	glEnableVertexAttribArray(h_nor);
	glBindBuffer(GL_ARRAY_BUFFER, norBufID);
	glBufferData(GL_ARRAY_BUFFER, norBuf.size() * sizeof(float), &norBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(h_nor, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	glDrawElements(GL_TRIANGLES, 3 * nTriFaces, GL_UNSIGNED_INT, (const void *)(0 * sizeof(unsigned int)));

	// Draw the AABB bounding box
	drawAABB();

	glDisableVertexAttribArray(h_nor);
	glDisableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	MV->popMatrix();
}


// Compute the AABB bounding box and save the index of vertices
void SPH::computeAABB() {
	max3 = Vector3f(INT_MIN + 0.01, INT_MIN + 0.01, INT_MIN + 0.01);
	min3 = Vector3f(INT_MAX - 0.01, INT_MAX - 0.01, INT_MAX - 0.01);
	index_max3.setZero();
	index_min3.setZero();

	for (int axis = 0; axis < 3; axis++) {
		for (int i = 0; i < nodes.size(); i++) {

			if (nodes[i]->x(axis) > max3(axis)) {
				max3(axis) = nodes[i]->x(axis);
				// save the node index for collisions
				index_max3(axis) = i;
			}

			if (nodes[i]->x(axis) < min3(axis)) {
				min3(axis) = nodes[i]->x(axis);
				// save the node index for collisions
				index_min3(axis) = i;
			}
		}
	}
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
		
		// check collision with 3d mesh
		int num = 0;

		for (int axis = 0; axis < 3; axis++) {
			if (pa->x(axis) > max3(axis) || pa->x(axis) < min3(axis)) {
				// no collision with mesh
			}
			else {
				num = num + 1;
			}
		}
		// if there is collsion in all three axies, they might collide with each other
		if (num == 3) {
			for (int iface = 0; iface < nTriFaces; iface++) {
				// face vertices
				Vector3f p1 = nodes[out.trifacelist[3 * iface + 0]]->x;
				Vector3f p2 = nodes[out.trifacelist[3 * iface + 1]]->x;
				Vector3f p3 = nodes[out.trifacelist[3 * iface + 2]]->x;

				Vector3f nor = -(p1 - p3).cross(p2 - p3);
				nor.normalize();
				
				Vector3f x_old = pa->x_old;
				Vector3f x_new = pa->x;
				Vector3f dir = x_new - x_old;
				float f = (x_old - p1).dot(nor) / ((x_old - p1).dot(nor) - (x_new - p1).dot(nor));
				Vector3f interpoint;
				interpoint.setZero();
				if (rayTriangleIntersects(p1, p2, p3, dir, x_old, interpoint) && f < 1.0) {
					// Collide with the face
					pa->x = interpoint + (interpoint - pa->x) *wallsticky;
					pa->v = - pa->v.dot(nor) * nor * wallsticky;
					
					//cout << "coll" << endl;
				}
			}
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
		pa->x_old = pa->x;
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
		pa->x_old = pa->x;
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
		pa->x_old = pa->x;
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

void SPH::drawAABB() const {
	glDisable(GL_LIGHTING);
	glLineWidth(1);
	glColor3f(0.5f, 0.5f, 0.5f);
	glBegin(GL_LINE_LOOP);
	glNormal3f(0, -1.0f, 0);
	glVertex3f(min3[0], min3[1], min3[2]); //0
	glVertex3f(max3[0], min3[1], min3[2]); //3
	glVertex3f(max3[0], min3[1], max3[2]); //2
	glVertex3f(min3[0], min3[1], max3[2]); //1
	glVertex3f(min3[0], max3[1], max3[2]); //5
	glVertex3f(max3[0], max3[1], max3[2]); //6
	glVertex3f(max3[0], max3[1], min3[2]); //7
	glVertex3f(min3[0], max3[1], min3[2]); //4
	glEnd();

	glBegin(GL_LINES);
	glNormal3f(0, -1.0f, 0);
	glVertex3f(min3[0], max3[1], min3[2]); //4
	glVertex3f(min3[0], max3[1], max3[2]); //5
	glVertex3f(min3[0], min3[1], min3[2]); //0
	glVertex3f(min3[0], min3[1], max3[2]); //1
	glVertex3f(max3[0], max3[1], max3[2]); //6
	glVertex3f(max3[0], min3[1], max3[2]); //2
	glVertex3f(max3[0], max3[1], min3[2]); //7
	glVertex3f(max3[0], min3[1], min3[2]); //3
	glEnd();
	glLineWidth(1);
	glEnable(GL_LIGHTING);
}

// This test function is adapted from Moller-Trumbore intersection algorithm. 
// See https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm 

bool SPH::rayTriangleIntersects(Vector3f v1, Vector3f v2, Vector3f v3, Vector3f dir, Vector3f pos, Vector3f &outpoint) {

	Vector3f e1 = v2 - v1;
	Vector3f e2 = v3 - v1;

	// Calculate planes normal vector
	//cross product
	Vector3f pvec = dir.cross(e2);

	//dot product
	float det = e1.dot(pvec);

	// Ray is parallel to plane
	if (det <1e-8 && det > -1e-8) {
		return false;
	}

	float inv_det = 1 / det;

	// Distance from v1 to ray pos
	Vector3f tvec = pos - v1;
	float u = (tvec.dot(pvec))*inv_det;
	if (u < 0 || u > 1) {
		return false;
	}

	Vector3f qvec = tvec.cross(e1);
	float v = dir.dot(qvec) * inv_det;
	if (v<0 || u + v>1) {
		return false;
	}

	float t = e2.dot(qvec) * inv_det;
	if (t > 1e-8) {
		outpoint = pos + dir * t;
		return true;
	}
	return false;
}


SPH::~SPH() {

}