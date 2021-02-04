
use rts::prelude::*;
use rts::system_mod::cont_circ::{ContCircSystem, ContCircSystemArguments};
use rts::target_mod::cont_bulk::{ContBulkTarget, ContBulkTargetArguments};
use rts::searcher_mod::{Passive, Interaction, cont_passive_exp::{ContPassiveExpSearcher, ContPassiveExpSearcherArguments}};
use rts::time_mod::{ExponentialStep, ExponentialStepArguments};

// Dataset
construct_dataset!(SimulationData, ContCircSystem, sys_arg, ContCircSystemArguments,
                [sys_size, f64, dim, usize ];
                ContBulkTarget, target_arg, ContBulkTargetArguments,
                [target_size, f64];
                ContPassiveExpSearcher, searcher_arg, ContPassiveExpSearcherArguments,
                [num_searcher, usize, gamma, f64, strength, f64];
                ExponentialStep, time_arg, ExponentialStepArguments,
                [dt_min, f64, dt_max, f64, length, usize];
                {Simulation, sim_arg, SimulationArguments,
                [idx_set, usize]});


fn main() -> Result<(), Error>{
    // System arguments : (sys_size) (dim)
    // Target arguments : (target_pos) (target_size)
    // Searcher arguments : (int_type) (mtype) (itype) (num_searcher)
    // Time Iterator arguments : (dt_min) (dt_max) (length) (tmax)
    // Simulation arguments : (num_ensemble) (idx_set) (seed) (output_dir)
    // let args : Vec<String> = ["10", "2", "0:0", "1", "Exponential(2, 1.0, 0.1)", "1.0", "Uniform", "1000", "1e-10", "1e-5", "10", "0", "100", "1", "12314123", "datas/benchmark"].iter().map(|x| x.to_string()).collect();
    setup_simulation!(args, 15, 1, MFPTAnalysis, "RTS_N_PTL_INTERACTING_SEARCHER", dataset, SimulationData,
        sys_arg, ContCircSystem, target_arg, ContBulkTarget,
        searcher_arg, ContPassiveExpSearcher, time_arg, ExponentialStep, sim_arg, Simulation);

    let sys_size    = sys_arg.sys_size;
    let dim         = sys_arg.dim;

    let _target_pos  = target_arg.target_pos.clone();
    let target_size = target_arg.target_size;

    let _mtype       = searcher_arg.mtype;
    let _itype       = searcher_arg.itype.clone();
    let _int_type    = searcher_arg.int_type;
    let num_searcher = searcher_arg.num_searcher;

    let _dt_min          = time_arg.dt_min;
    let _dt_max          = time_arg.dt_max;
    let _length          = time_arg.length;

    let num_ensemble= sim_arg.num_ensemble;
    let idx_set     = sim_arg.idx_set;
    let seed        = sim_arg.seed;
    let output_dir  = sim_arg.output_dir.clone();

    // Hash seed and generate random number generator
    let seed : u128 = seed + (628_398_227f64 * sys_size +
                              431_710_567f64 * dim as f64 +
                              277_627_711f64 * target_size +
                              719_236_607f64 * num_searcher as f64 +

                              570_914_867f64 * idx_set as f64).floor() as u128;
    let mut rng : Pcg64 = rng_seed(seed);

    // System initiation
    export_simulation_info!(dataset, output_dir, writer, WIDTH, "RTS_N_PTL_MERGEABLE_SEARCHER",
                            ContCircSystem, sys, sys_arg,
                            ContBulkTarget, target, target_arg,
                            ContPassiveExpSearcher, vec_searchers, searcher_arg,
                            ExponentialStep, timeiter, time_arg,
                            Simulation, simulation, sim_arg);

    let mut single_move = vec![Position::<f64>::new(vec![0f64; dim]); num_searcher];
    let mut list_searchers : LinkedList<ContPassiveExpSearcher> = LinkedList::from(vec_searchers);

    let mut disp = Position::<f64>::new(vec![0f64; dim]);

    for _i in 0..num_ensemble{
        let mut fpt : f64 = 0f64;

        for s in &mut list_searchers.contents{
            s.renew_uniform(&sys, &target, &mut rng)?;
        }
        list_searchers.connect_all()?;

        'outer : for (time, dt) in timeiter.into_diff().skip(1){

            // clear temporal vector for infinitesimal displacement
            // compute single random walk
            list_searchers.into_iter();
            while let Some((idx, searcher)) = list_searchers.enumerate_mut(){
                single_move[idx].clear();
                searcher.random_move_to_vec(&mut rng, dt, &mut single_move[idx])?;
            }

            // compute force
            list_searchers.into_double_iter();
            while let Some((idx1, s1, idx2, s2)) = list_searchers.enumerate_double(){
                let r : f64 = s1.mutual_displacement_to_vec(&s2, &mut disp)?;
                let force = s1.force(r);

                disp.mut_scalar_mul(force * dt);
                single_move[idx1].mut_add(&disp)?;
                disp.mut_scalar_mul(-1f64);
                single_move[idx2].mut_add(&disp)?;
            }

            list_searchers.into_iter();
            while let Some((idx, searcher)) = list_searchers.enumerate_mut(){
                sys.check_bc(&mut searcher.pos, &mut single_move[idx])?;
                if target.check_find(&searcher.pos)?{
                    fpt = time;
                    break 'outer;
                }
            }
        }

        // Export FPT data
        write!(&mut writer, "{0:.5e}\n", fpt).map_err(Error::make_error_io)?;
        writer.flush().map_err(Error::make_error_io)?;
    }

    return Ok(());
}
