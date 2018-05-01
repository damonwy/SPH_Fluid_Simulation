#pragma once

#ifndef __Node__
#define __Node__

#include <vector>
#include <memory>

#define EIGEN_DONT_ALIGN_STATICALLY
#include <Eigen/Dense>

class Shape;
class Program;
class MatrixStack;

class Node
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		Node();
	Node(const std::shared_ptr<Shape> shape);
	virtual ~Node();
	void tare();
	void reset();
	void draw(std::shared_ptr<MatrixStack> MV, const std::shared_ptr<Program> p) const;

	float r; // radius
	float m; // mass
	int i;  // starting index
	Eigen::Vector3f x0; // initial position
	Eigen::Vector3f v0; // initial velocity
	Eigen::Vector3f x_old;
	Eigen::Vector3f x;  // position
	Eigen::Vector3f v;  // velocity
	Eigen::Vector3f normal;

	float lifespan;
	float tEnd;
	float alpha;
	float scale;
	Eigen::Vector3f color;

	bool fixed;

private:
	const std::shared_ptr<Shape> sphere;
};

#endif