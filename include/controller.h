#pragma once
#ifndef __CONTROLLER_H
#define __CONTROLLER_H

#include <iostream>
#include <Eigen/Dense>
#include <rbdl/rbdl.h>
#include <rbdl/addons/urdfreader/urdfreader.h>

#include "robotmodel.h"
#include "trajectory.h"
#include "custommath.h"

using namespace std;
using namespace Eigen;

#define NECS2SEC 1000000000

class CController
{

public:
    CController();
    virtual ~CController();	

    void read(double time, double* q, double* qdot);
    void control_mujoco();
    void write(double* torque);

    void read_pybind(double time, std::array<double,7> qpos, std::array<double, 7> qvel);
    std::vector<double> write_pybind();
    std::vector<double> torque_command;

private:
    void Initialize();
    void ModelUpdate();
    void motionPlan();

    void reset_target(double motion_time, VectorXd target_joint_position);
    void reset_target(double motion_time, VectorXd target_joint_position, VectorXd target_joint_velocity);
    void reset_target(double motion_time, Vector3d target_pos, Vector3d target_ori);
    
    VectorXd _q; // joint angle
	VectorXd _qdot; // joint velocity
    VectorXd _torque; // joint torque

    int _k; // DOF

    bool _bool_init;
    double _t;
    double _dt;
	double _init_t;
	double _pre_t;

    //controller
	double _kpj, _kdj; //joint P,D gain
    double _x_kp; // task control P gain

    void JointControl();
    void CLIK();

    // robotmodel
    CModel Model;

    int _cnt_plan;
	VectorXd _time_plan;
	VectorXi _bool_plan;

    int _control_mode; //1: joint space, 2: operational space
    VectorXd _q_home; // joint home position

    //motion trajectory
	double _start_time, _end_time, _motion_time;

    CTrajectory JointTrajectory; // joint space trajectory
    HTrajectory HandTrajectory; // task space trajectory

    bool _bool_joint_motion, _bool_ee_motion; // motion check

    VectorXd _q_des, _qdot_des; 
    VectorXd _q_goal, _qdot_goal;
    VectorXd _x_des_hand, _xdot_des_hand;
    VectorXd _x_goal_hand, _xdot_goal_hand;
    Vector3d _pos_goal_hand, _rpy_goal_hand;

    MatrixXd _A_diagonal; // diagonal inertia matrix
    MatrixXd _J_hands; // jacobian matrix
    MatrixXd _J_bar_hands; // pseudo invere jacobian matrix

    VectorXd _x_hand, _xdot_hand; // End-effector


    VectorXd _x_err_hand;
    Matrix3d _R_des_hand;

    MatrixXd _I; // Identity matrix
};

#endif
