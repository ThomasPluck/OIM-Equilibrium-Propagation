def log_metrics_to_wandb(all_metrics, model_seeds, args):
    """Log loaded metrics to wandb"""
    if args.wandb_mode == 'disabled':
        print("Wandb logging is disabled")
        return
    
    for model_idx, metrics_list in all_metrics.items():
        if not metrics_list:
            print(f"No metrics to log for model {model_idx}")
            continue
            
        print(f"Logging {len(metrics_list)} metrics for model {model_idx}...")
        
        # Clean up name and group - remove any quotes
        name_base = args.wandb_name or 'run'
        group_base = args.wandb_group or 'parallel-group'
        
        # Remove quotes and backslashes
        name_base = name_base.replace('"', '').replace("'", '').replace('\\', '')
        group_base = group_base.replace('"', '').replace("'", '').replace('\\', '')
        
        # Initialize wandb run for this model
        run_name = f"{name_base}-model{model_idx}"
        group_name = f"{group_base}"
        
        # Create config with model index and seed
        config = {
            'model_index': model_idx,
        }
        if model_idx in model_seeds:
            config['seed'] = model_seeds[model_idx]
        
        # Initialize wandb run
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=group_name,
            config=config,
            mode=args.wandb_mode,
            reinit=True,  # Need this since we're creating multiple runs
        )
        
        try:
            # Sort metrics by epoch to ensure correct order
            sorted_metrics = sorted(metrics_list, key=lambda x: x['epoch'])
            max_epoch = 0
            
            # Log each metric with correct epoch
            for metric_item in sorted_metrics:
                epoch = metric_item['epoch']
                metrics = metric_item['metrics']
                
                # Update max epoch seen
                max_epoch = max(max_epoch, epoch)
                
                # Log metrics
                run.log(metrics, step=epoch)
            
            # Log completion metrics
            run.log({
                "training_completed": False,  # Set to False since it was interrupted
                "last_logged_epoch": max_epoch
            })
            
            print(f"Successfully logged metrics for model {model_idx} to WandB")
        except Exception as e:
            print(f"Error logging metrics for model {model_idx} to WandB: {e}")
        finally:
            # Finish the run
            run.finish() 