#ifndef _INTEGRATOR_WHFAST_HJ_H
#define _INTEGRATOR_WHFAST_HJ_H

extern const struct reb_integrator reb_integrator_whfast_hj;
REB_API int reb_integrator_whfast_hj_tree_to_string(struct reb_simulation* const r, char* const buffer, const size_t buffer_size);
REB_API int reb_integrator_whfast_hj_set_tree(struct reb_simulation* const r, const char* const tree);
REB_API int reb_integrator_whfast_hj_set_binary_plus_particles_tree(struct reb_simulation* const r);
REB_API void reb_integrator_whfast_hj_clear_tree(struct reb_simulation* const r);

struct reb_integrator_whfast_hj_node {
    struct reb_particle barycenter_particle;
    struct reb_particle jacobi_particle;

    size_t primary;
    size_t secondary;

    double primary_offset;
    double secondary_offset;
};

struct reb_integrator_whfast_hj_state {
    // Fixed hierarchy, stored in postorder: leaves first, then binary orbits.
    struct reb_integrator_whfast_hj_node* nodes;
    size_t tree_N;

    // Set after a user-supplied tree has been compiled into nodes.
    int given_tree;
};

#endif
