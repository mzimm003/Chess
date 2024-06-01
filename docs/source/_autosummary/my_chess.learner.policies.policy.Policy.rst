my\_chess.learner.policies.policy.Policy
========================================

.. currentmodule:: my_chess.learner.policies.policy

.. autoclass:: Policy
   :members:
   :show-inheritance:
   :inherited-members:

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~Policy.__init__
      ~Policy.action_distribution_fn
      ~Policy.action_sampler_fn
      ~Policy.apply
      ~Policy.apply_gradients
      ~Policy.compute_actions
      ~Policy.compute_actions_from_input_dict
      ~Policy.compute_gradients
      ~Policy.compute_log_likelihoods
      ~Policy.compute_single_action
      ~Policy.export_checkpoint
      ~Policy.export_model
      ~Policy.extra_action_out
      ~Policy.extra_compute_grad_fetches
      ~Policy.extra_grad_process
      ~Policy.from_checkpoint
      ~Policy.from_state
      ~Policy.get_batch_divisibility_req
      ~Policy.get_connector_metrics
      ~Policy.get_exploration_info
      ~Policy.get_exploration_state
      ~Policy.get_host
      ~Policy.get_initial_state
      ~Policy.get_num_samples_loaded_into_buffer
      ~Policy.get_session
      ~Policy.get_state
      ~Policy.get_tower_stats
      ~Policy.get_weights
      ~Policy.import_model_from_h5
      ~Policy.init_view_requirements
      ~Policy.is_recurrent
      ~Policy.learn_on_batch
      ~Policy.learn_on_batch_from_replay_buffer
      ~Policy.learn_on_loaded_batch
      ~Policy.load_batch_into_buffer
      ~Policy.loss
      ~Policy.loss_initialized
      ~Policy.make_model
      ~Policy.make_model_and_action_dist
      ~Policy.make_rl_module
      ~Policy.maybe_add_time_dimension
      ~Policy.maybe_remove_time_dimension
      ~Policy.num_state_tensors
      ~Policy.on_global_var_update
      ~Policy.optimizer
      ~Policy.postprocess_trajectory
      ~Policy.reset_connectors
      ~Policy.restore_connectors
      ~Policy.set_state
      ~Policy.set_weights
      ~Policy.stats_fn
   
   

   
   
   