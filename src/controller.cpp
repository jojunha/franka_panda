#include "controller.h"
#include <chrono>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream> // ifstream header
#include <iostream>
#include <string> // getline header

CController::CController()
{
	_k = 7;
	Initialize();
}

CController::~CController()
{
}

void CController::read(double t, double* q, double* qdot)
{	
	_t = t;
	if (_bool_init == true)
	{
		_init_t = _t;
		_bool_init = false;
	}

	_dt = t - _pre_t;
	_pre_t = t;

	for (int i = 0; i < _k; i++)
	{
		_q(i) = q[i];
		_qdot(i) = qdot[i];		
	}
}

void CController::write(double* torque)
{
	for (int i = 0; i < _k; i++)
	{
		torque[i] = _torque(i);
	}
}


// for pybind11
////////////////////////////////////////////////////////////////////////////////////////////////
void CController::read_pybind(double t, std::array<double,7> q, std::array<double, 7> qdot)
{	
	_t = t;
	if (_bool_init == true)
	{
		_init_t = _t;
		_bool_init = false;
	}

	_dt = t - _pre_t;
	_pre_t = t;

	for (int i = 0; i < _k; i++)
	{
		_q(i) = q[i];
		_qdot(i) = qdot[i];		
	}
}

std::vector<double> CController::write_pybind()
{
	torque_command.clear();

	for (int i = 0; i < _k; i++)
	{
		torque_command.push_back(_torque(i));
	}

	return torque_command;
}
////////////////////////////////////////////////////////////////////////////////////////////////

void CController::control_mujoco()
{
    ModelUpdate();
    motionPlan();

	if(_control_mode == 1) // joint space control
	{
		if (_t - _init_t < 0.1 && _bool_joint_motion == false)
		{
			_start_time = _init_t;
			_end_time = _start_time + _motion_time;
			JointTrajectory.reset_initial(_start_time, _q, _qdot);
			JointTrajectory.update_goal(_q_goal, _qdot_goal, _end_time);
			_bool_joint_motion = true;
		}

		JointTrajectory.update_time(_t);

		_q_des = JointTrajectory.position_cubicSpline();
		_qdot_des = JointTrajectory.velocity_cubicSpline();


		JointControl();

		if (JointTrajectory.check_trajectory_complete() == 1)
		{
			_bool_plan(_cnt_plan) = 1;
			_bool_init = true;
		}
	}
	else if(_control_mode == 2) // task space control
	{
		if (_t - _init_t < 0.1 && _bool_ee_motion == false)
		{
			_start_time = _init_t;
			_end_time = _start_time + _motion_time;
			HandTrajectory.reset_initial(_start_time, _x_hand, _xdot_hand);
			HandTrajectory.update_goal(_x_goal_hand, _xdot_goal_hand, _end_time); 
			_bool_ee_motion = true;
		}

		HandTrajectory.update_time(_t);

		_x_des_hand.head(3) = HandTrajectory.position_cubicSpline();
		_R_des_hand = HandTrajectory.rotationCubic();
		_x_des_hand.segment<3>(3) = CustomMath::GetBodyRotationAngle(_R_des_hand);

		_xdot_des_hand.head(3) = HandTrajectory.velocity_cubicSpline();
		_xdot_des_hand.segment<3>(3) = HandTrajectory.rotationCubicDot();

		CLIK();

		if (HandTrajectory.check_trajectory_complete() == 1)
		{
			_bool_plan(_cnt_plan) = 1;
			_bool_init = true;
		}
	}
}



void CController::ModelUpdate()
{
    Model.update_kinematics(_q, _qdot);
	Model.update_dynamics();
    Model.calculate_EE_Jacobians();
	Model.calculate_EE_positions_orientations();
	Model.calculate_EE_velocity();

	_J_hands = Model._J_hand;

	_x_hand.head(3) = Model._x_hand;
	_x_hand.tail(3) = CustomMath::GetBodyRotationAngle(Model._R_hand);

	_xdot_hand = Model._xdot_hand;
}

void CController::motionPlan()
{	
	_time_plan(1) = 2.0; // move home position
	_time_plan(2) = 1.0; // wait
	_time_plan(3) = 2.0; // joint goal motion
	_time_plan(4) = 1.0; // wait
	_time_plan(5) = 2.0; // task goal motion
	_time_plan(6) = 100000.0; // wait

	if (_bool_plan(_cnt_plan) == 1)
	{
		_cnt_plan = _cnt_plan + 1;

		if(_cnt_plan == 1)
		{	
			reset_target(_time_plan(_cnt_plan), _q_home);
		}
		else if (_cnt_plan == 2)
		{
			reset_target(_time_plan(_cnt_plan), _q);
		}
		else if (_cnt_plan == 3)
		{	
			_q_goal.setZero(_k);
			_q_goal(0) = 0.0;
			_q_goal(1) = 0.0; 
			_q_goal(2) = 0.0; 
			_q_goal(3) = -90.0 * DEG2RAD; 
			_q_goal(4) = 0.0; 
			_q_goal(5) = 90 * DEG2RAD; 
			_q_goal(6) = 0.0; 

			reset_target(_time_plan(_cnt_plan), _q_goal);
		}
		else if (_cnt_plan == 4)
		{
			reset_target(_time_plan(_cnt_plan), _q);
		}
		else if (_cnt_plan == 5)
		{
			_pos_goal_hand(0) = _x_hand(0) - 0.2;
			_pos_goal_hand(1) = _x_hand(1) + 0.2;
			_pos_goal_hand(2) = _x_hand(2);

			_rpy_goal_hand(0) = _x_hand(3);
			_rpy_goal_hand(1) = _x_hand(4) - 0.2;
			_rpy_goal_hand(2) = _x_hand(5) + 0.5;

			reset_target(_time_plan(_cnt_plan), _pos_goal_hand, _rpy_goal_hand);
		}
		else if (_cnt_plan == 6)
		{
			reset_target(_time_plan(_cnt_plan), _q);
		}	
	}
}


void CController::reset_target(double motion_time, VectorXd target_joint_position)
{
	_control_mode = 1;
	_motion_time = motion_time;
	_bool_joint_motion = false;
	_bool_ee_motion = false;

	_q_goal = target_joint_position.head(7);
	_qdot_goal.setZero();
}

void CController::reset_target(double motion_time, VectorXd target_joint_position, VectorXd target_joint_velocity)
{
	_control_mode = 1;
	_motion_time = motion_time;
	_bool_joint_motion = false;
	_bool_ee_motion = false;

	_q_goal = target_joint_position.head(7);
	_qdot_goal = target_joint_velocity.head(7);
}

void CController::reset_target(double motion_time, Vector3d target_pos, Vector3d target_ori)
{
	_control_mode = 2;
	_motion_time = motion_time;
	_bool_joint_motion = false;
	_bool_ee_motion = false;

	_x_goal_hand.head(3) = target_pos;
	_x_goal_hand.tail(3) = target_ori;
	_xdot_goal_hand.setZero();
}

void CController::JointControl()
{	
	_torque.setZero();
	_torque = Model._A*(_kpj*(_q_des - _q) + _kdj*(_qdot_des - _qdot)) + Model._bg;
}

// Closed Loop Inverse Kinematics
void CController::CLIK()
{
	_torque.setZero();	

	_x_err_hand.segment(0,3) = _x_des_hand.head(3) - _x_hand.head(3);
	_x_err_hand.segment(3,3) = -CustomMath::getPhi(Model._R_hand, _R_des_hand);

	_J_bar_hands = CustomMath::pseudoInverseQR(_J_hands);
	// _J_bar_hands = CustomMath::DampedWeightedPseudoInverse(_J_hands,_I*0.01,true);


	_qdot_des = _J_bar_hands*(_xdot_des_hand + _x_kp*(_x_err_hand));
	_q_des = _q + _dt*_qdot_des;


	for(int i = 0; i < 7; i++){
		_A_diagonal(i,i) = Model._A(i,i);
	}

	_torque = _A_diagonal*(_kpj*(_q_des - _q) + _kdj*(_qdot_des - _qdot)) + Model._bg;

	// cout << _torque.transpose() << endl;
	// _torque =  Model._A*(_kp*(_q_des - _q) + _kd*(_qdot_des - _qdot)) + Model._bg;
}

void CController::Initialize()
{
    _control_mode = 1; //1: joint space, 2: task space(CLIK)

	_bool_init = true;
	_t = 0.0;
	_init_t = 0.0;
	_pre_t = 0.0;
	_dt = 0.0;

	_kpj = 400.0;
	_kdj = 40.0;

	_x_kp = 20.0;

    _q.setZero(_k);
	_qdot.setZero(_k);
	_torque.setZero(_k);

	_J_hands.setZero(6,_k);
	_J_bar_hands.setZero(_k,6);

	_x_hand.setZero(6);
	_xdot_hand.setZero(6);

	_cnt_plan = 0;
	_bool_plan.setZero(30);
	_time_plan.resize(30);
	_time_plan.setConstant(5.0);

	_q_home.setZero(_k);
	_q_home(0) = 0.0;
	_q_home(1) = -30.0 * DEG2RAD; 
	_q_home(2) = 30.0 * DEG2RAD; 
	_q_home(3) = -30.0 * DEG2RAD; 
	_q_home(4) = 30.0 * DEG2RAD; 
	_q_home(5) = -60.0 * DEG2RAD; 
	_q_home(6) = 30.0 * DEG2RAD; 

	_start_time = 0.0;
	_end_time = 0.0;
	_motion_time = 0.0;

	_bool_joint_motion = false;
	_bool_ee_motion = false;

	_q_des.setZero(_k);
	_qdot_des.setZero(_k);
	_q_goal.setZero(_k);
	_qdot_goal.setZero(_k);

	_x_des_hand.setZero(6);
	_xdot_des_hand.setZero(6);
	_x_goal_hand.setZero(6);
	_xdot_goal_hand.setZero(6);

	_pos_goal_hand.setZero(); // 3x1 
	_rpy_goal_hand.setZero(); // 3x1

	JointTrajectory.set_size(_k);
	_A_diagonal.setZero(_k,_k);

	torque_command.clear();

	_x_err_hand.setZero(6);
	_R_des_hand.setZero();

	_I.setIdentity(7,7);
}



namespace py = pybind11;
PYBIND11_MODULE(controller, m)
{
  m.doc() = "pybind11 for controller";

  py::class_<CController>(m, "CController")
      .def(py::init())
      .def("read", &CController::read_pybind)
	  .def("control_mujoco", &CController::control_mujoco)
	  .def("write", &CController::write_pybind);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
//   m.attr("TEST") = py::int_(int(42));
}
