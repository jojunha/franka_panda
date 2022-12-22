#include "robotmodel.h"
#define JDOF 7

CModel::CModel()
{
	Initialize();
}

CModel::~CModel()
{
}

void CModel::Initialize()
{
	_bool_model_update = false;
	_bool_kinematics_update = false;
	_bool_dynamics_update = false;
	_bool_Jacobian_update = false;

    _k = JDOF;
    _id_hand = 7;

	_max_joint_torque.setZero(_k);
	_min_joint_torque.setZero(_k);
	_max_joint_velocity.setZero(_k);
	_min_joint_velocity.setZero(_k);
	_max_joint_position.setZero(_k);
	_min_joint_position.setZero(_k);

    _q.setZero(_k);
    _qdot.setZero(_k);
    _zero_vec_joint.setZero(_k);

    _A.setZero(_k,_k);
    _g.setZero(_k);
	_b.setZero(_k);
	_bg.setZero(_k);

    _J_hand.setZero(6,_k);
    _J_tmp.setZero(6,_k);

    _position_local_task_hand.setZero(); // 3x1
    _x_hand.setZero(); // 3x1
    _R_hand.setZero(); // 3x3

    _xdot_hand.setZero(6);

	set_robot_config();
    load_model();
}


void CModel::load_model()
{   
    RigidBodyDynamics::Addons::URDFReadFromFile("/home/kist-robot2/catkin_ws/src/franka_panda/model/panda.urdf", &_model, false, true);

    cout << endl << endl << "Model Loaded for RBDL." << endl << "Total DoFs: " << _model.dof_count << endl << endl;
	if (_model.dof_count != _k)
	{
		cout << "Simulation model and RBDL model mismatch!!!" << endl << endl;
	}

    _bool_model_update = true; //check model update

	cout << "Model Loading Complete." << endl << endl;

}

void CModel::update_kinematics(VectorXd & q, VectorXd & qdot)
{
	_q = q;
	_qdot = qdot;

	if (_bool_model_update == true)
	{
		RigidBodyDynamics::UpdateKinematicsCustom(_model, &_q, &_qdot, NULL); // update kinematics
	}
	else
	{
		cout << "Robot model is not ready. Please load model first." << endl << endl;
	}
	_bool_kinematics_update = true; // check kinematics update
}

void CModel::update_dynamics()
{
	if (_bool_kinematics_update == true)
	{	
		RigidBodyDynamics::CompositeRigidBodyAlgorithm(_model, _q, _A, false); // update inertia matrix
		RigidBodyDynamics::InverseDynamics(_model, _q, _zero_vec_joint, _zero_vec_joint, _g, NULL); // get _g
		RigidBodyDynamics::InverseDynamics(_model, _q, _qdot, _zero_vec_joint, _bg, NULL); // get _g+_b
		_b = _bg - _g; //get _b
	}
	else
	{
		cout << "Robot kinematics is not ready. Please update kinematics first." << endl << endl;
	}
	_bool_dynamics_update = true; // check kinematics update
}

void CModel::calculate_EE_Jacobians()
{
	if (_bool_kinematics_update == true)
	{
		_J_hand.setZero();
		_J_tmp.setZero();	

		RigidBodyDynamics::CalcPointJacobian6D(_model, _q, _id_hand, _position_local_task_hand, _J_tmp, false); // update kinematc : false
		_J_hand.block<3, 7>(0, 0) = _J_tmp.block<3, 7>(3, 0); // linear : last three entries -> first three entries
		_J_hand.block<3, 7>(3, 0) = _J_tmp.block<3, 7>(0, 0); // angular : first three entries -> last three entries

		_bool_Jacobian_update = true;
	}
	else
	{
		cout << "Robot kinematics is not ready. Please update kinematics first." << endl << endl;
	}

}

void CModel::calculate_EE_positions_orientations()
{
    if (_bool_kinematics_update == true)
	{
		_x_hand.setZero();
		_R_hand.setZero();

		_x_hand = RigidBodyDynamics::CalcBodyToBaseCoordinates(_model, _q, _id_hand, _position_local_task_hand, false);
		_R_hand = RigidBodyDynamics::CalcBodyWorldOrientation(_model, _q, _id_hand, false).transpose();

	}
	else
	{
		cout << "Robot kinematics is not ready. Please update kinematics first." << endl << endl;
	}
}

void CModel::calculate_EE_velocity()
{
	if (_bool_Jacobian_update == true)
	{
		_xdot_hand = _J_hand * _qdot;
	}
	else
	{
		cout << "Jacobian matrices are not ready. Please calculate Jacobians first." << endl << endl;
	}
}

void CModel::set_robot_config(){
	_max_joint_position(0) = 166.0 * DEG2RAD;
	_min_joint_position(0) = -166.0 * DEG2RAD;
	_max_joint_position(1) = 101.0 * DEG2RAD;
	_min_joint_position(1) = -101.0 * DEG2RAD;
	_max_joint_position(2) = 166.0 * DEG2RAD;
	_min_joint_position(2) = -166.0 * DEG2RAD;
	_max_joint_position(3) = 176.0 * DEG2RAD;
	_min_joint_position(3) = -4.0 * DEG2RAD;
	_max_joint_position(4) = 166.0 * DEG2RAD;
	_min_joint_position(4) = -166.0 * DEG2RAD;
	_max_joint_position(5) = 215.0 * DEG2RAD;
	_min_joint_position(5) = -1.0 * DEG2RAD;
	_max_joint_position(6) = 166.0 * DEG2RAD;
	_min_joint_position(6) = -166.0 * DEG2RAD;

	_max_joint_velocity(0) = 155.0 * DEG2RAD;
	_max_joint_velocity(1) = 150.0 * DEG2RAD;
	_max_joint_velocity(2) = 150.0 * DEG2RAD;
	_max_joint_velocity(3) = 150.0 * DEG2RAD;
	_max_joint_velocity(4) = 180.0 * DEG2RAD;
	_max_joint_velocity(5) = 180.0 * DEG2RAD;
	_max_joint_velocity(6) = 180.0 * DEG2RAD;
	_min_joint_velocity = -_max_joint_velocity;

}
