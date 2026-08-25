#include "rebound.h"
#include "rebound_internal.h"
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "integrator_whfast.h"
#include "integrator_whfast_HJ.h"
#include "binarydata.h"

void* reb_integrator_whfast_hj_create(void);
void reb_integrator_whfast_hj_free(void* state);
void reb_integrator_whfast_hj_step(struct reb_simulation* const r, void* state);
const struct reb_binarydata_field_descriptor reb_integrator_whfast_hj_field_descriptor_list[];

const struct reb_integrator reb_integrator_whfast_hj = {
    .documentation =
    "WHFast HJ is a hierarchical Jacobi-coordinate variant of WHFast. "
    "It requires a fixed user-supplied binary hierarchy, which is compiled once "
    "into a flat postordered representation.",
    .step = reb_integrator_whfast_hj_step,
    .create = reb_integrator_whfast_hj_create,
    .free = reb_integrator_whfast_hj_free,
    .field_descriptor_list = reb_integrator_whfast_hj_field_descriptor_list,
};

const struct reb_binarydata_field_descriptor reb_integrator_whfast_hj_field_descriptor_list[] = {
    { "Whether a fixed user-supplied HJ tree has been configured.",
        REB_INT, "given_tree", offsetof(struct reb_integrator_whfast_hj_state, given_tree), 0, 0, 0},
    { 0 },
};

static size_t reb_integrator_whfast_hj_node_count(const size_t N)
{
    return N == 0 ? 0 : 2*N - 1;
}

void* reb_integrator_whfast_hj_create(void)
{
    return calloc(1, sizeof(struct reb_integrator_whfast_hj_state));
}

static void reb_integrator_whfast_hj_clear_state(struct reb_integrator_whfast_hj_state* const whfast)
{
    free(whfast->nodes);
    whfast->nodes = NULL;
    whfast->tree_N = 0;
    whfast->given_tree = 0;
}

void reb_integrator_whfast_hj_free(void* state)
{
    struct reb_integrator_whfast_hj_state* const whfast = state;
    if (whfast == NULL){
        return;
    }
    reb_integrator_whfast_hj_clear_state(whfast);
    free(whfast);
}

static void reb_integrator_whfast_hj_initialize_leaf(
    struct reb_integrator_whfast_hj_node* const node,
    const double mass
){
    *node = (struct reb_integrator_whfast_hj_node){0};
    node->primary = SIZE_MAX;
    node->secondary = SIZE_MAX;
    node->barycenter_particle.m = mass;
}

static void reb_integrator_whfast_hj_update_internal_coordinate(
    struct reb_integrator_whfast_hj_state* const whfast,
    const size_t index
){
    struct reb_integrator_whfast_hj_node* const node = &whfast->nodes[index];
    const struct reb_particle primary = whfast->nodes[node->primary].barycenter_particle;
    const struct reb_particle secondary = whfast->nodes[node->secondary].barycenter_particle;

    node->barycenter_particle = reb_particle_com_of_pair(primary, secondary);

    node->jacobi_particle = (struct reb_particle){0};
    node->jacobi_particle.x = secondary.x - primary.x;
    node->jacobi_particle.y = secondary.y - primary.y;
    node->jacobi_particle.z = secondary.z - primary.z;
    node->jacobi_particle.vx = secondary.vx - primary.vx;
    node->jacobi_particle.vy = secondary.vy - primary.vy;
    node->jacobi_particle.vz = secondary.vz - primary.vz;
    node->jacobi_particle.ax = secondary.ax - primary.ax;
    node->jacobi_particle.ay = secondary.ay - primary.ay;
    node->jacobi_particle.az = secondary.az - primary.az;
}

static int reb_integrator_whfast_hj_initialize_internal(
    struct reb_integrator_whfast_hj_state* const whfast,
    const size_t index,
    const size_t primary,
    const size_t secondary
){
    struct reb_integrator_whfast_hj_node* const node = &whfast->nodes[index];
    const double primary_mass = whfast->nodes[primary].barycenter_particle.m;
    const double secondary_mass = whfast->nodes[secondary].barycenter_particle.m;
    const double total_mass = primary_mass + secondary_mass;

    *node = (struct reb_integrator_whfast_hj_node){0};
    node->primary = primary;
    node->secondary = secondary;
    node->barycenter_particle.m = total_mass;
    if (!(total_mass > 0.)){
        return 1;
    }
    node->primary_offset = secondary_mass/total_mass;
    node->secondary_offset = primary_mass/total_mass;
    return 0;
}

struct reb_integrator_whfast_hj_tree_parser {
    struct reb_simulation* r;
    struct reb_integrator_whfast_hj_state* whfast;
    const char* cursor;
    unsigned char* used;
    size_t leaf_count;
    size_t next_internal;
    const char* error;
};

static void reb_integrator_whfast_hj_tree_parser_skip_space(struct reb_integrator_whfast_hj_tree_parser* const parser)
{
    while (isspace((unsigned char)*parser->cursor)){
        parser->cursor++;
    }
}

static void reb_integrator_whfast_hj_tree_parser_error(
    struct reb_integrator_whfast_hj_tree_parser* const parser,
    const char* const error
){
    if (parser->error == NULL){
        parser->error = error;
    }
}

// The input grammar is recursive, but the result is a flat postordered array.
// Recursion is used only while compiling a user-supplied tree, never per timestep.
static size_t reb_integrator_whfast_hj_parse_tree_node(struct reb_integrator_whfast_hj_tree_parser* const parser)
{
    reb_integrator_whfast_hj_tree_parser_skip_space(parser);

    if (*parser->cursor == '['){
        parser->cursor++;
        reb_integrator_whfast_hj_tree_parser_skip_space(parser);
        if (*parser->cursor == ']'){
            parser->cursor++;
            return SIZE_MAX;
        }

        const size_t primary = reb_integrator_whfast_hj_parse_tree_node(parser);
        if (parser->error != NULL){
            return SIZE_MAX;
        }
        if (primary == SIZE_MAX){
            reb_integrator_whfast_hj_tree_parser_error(parser, "Invalid WHFast HJ tree: empty subtree.");
            return SIZE_MAX;
        }

        reb_integrator_whfast_hj_tree_parser_skip_space(parser);
        if (*parser->cursor != ','){
            reb_integrator_whfast_hj_tree_parser_error(parser, "Invalid WHFast HJ tree: expected ','.");
            return SIZE_MAX;
        }
        parser->cursor++;

        const size_t secondary = reb_integrator_whfast_hj_parse_tree_node(parser);
        if (parser->error != NULL){
            return SIZE_MAX;
        }
        if (secondary == SIZE_MAX){
            reb_integrator_whfast_hj_tree_parser_error(parser, "Invalid WHFast HJ tree: empty subtree.");
            return SIZE_MAX;
        }

        reb_integrator_whfast_hj_tree_parser_skip_space(parser);
        if (*parser->cursor != ']'){
            reb_integrator_whfast_hj_tree_parser_error(parser, "Invalid WHFast HJ tree: expected ']'.");
            return SIZE_MAX;
        }
        parser->cursor++;

        if (parser->next_internal >= reb_integrator_whfast_hj_node_count(parser->whfast->tree_N)){
            reb_integrator_whfast_hj_tree_parser_error(parser, "Invalid WHFast HJ tree: too many binary nodes.");
            return SIZE_MAX;
        }
        const size_t index = parser->next_internal++;
        if (reb_integrator_whfast_hj_initialize_internal(parser->whfast, index, primary, secondary)){
            reb_integrator_whfast_hj_tree_parser_error(parser, "Invalid WHFast HJ tree: each binary orbit must have positive total mass.");
            return SIZE_MAX;
        }
        return index;
    }

    if (isdigit((unsigned char)*parser->cursor)){
        unsigned long long particle_number = 0;
        while (isdigit((unsigned char)*parser->cursor)){
            const unsigned int digit = (unsigned int)(*parser->cursor - '0');
            if (particle_number > (ULLONG_MAX - digit)/10ULL){
                reb_integrator_whfast_hj_tree_parser_error(parser, "Invalid WHFast HJ tree: particle index is too large.");
                return SIZE_MAX;
            }
            particle_number = 10ULL*particle_number + digit;
            parser->cursor++;
        }

        if (particle_number == 0ULL || particle_number > (unsigned long long)parser->r->N){
            reb_integrator_whfast_hj_tree_parser_error(parser, "Invalid WHFast HJ tree: particle index out of range.");
            return SIZE_MAX;
        }
        const size_t particle_index = (size_t)(particle_number - 1ULL);
        if (parser->used[particle_index]){
            reb_integrator_whfast_hj_tree_parser_error(parser, "Invalid WHFast HJ tree: duplicate particle index.");
            return SIZE_MAX;
        }

        parser->used[particle_index] = 1;
        parser->leaf_count++;
        reb_integrator_whfast_hj_initialize_leaf(
            &parser->whfast->nodes[particle_index],
            parser->r->particles[particle_index].m
        );
        return particle_index;
    }

    reb_integrator_whfast_hj_tree_parser_error(parser, "Invalid WHFast HJ tree: expected '[' or particle index.");
    return SIZE_MAX;
}

static int reb_integrator_whfast_hj_allocate_fixed_tree(
    struct reb_simulation* const r,
    struct reb_integrator_whfast_hj_state* const whfast
){
    if (r->N > 0 && r->N > SIZE_MAX/2U + 1U){
        reb_simulation_error(r, "WHFast HJ tree does not support this many particles.");
        return 1;
    }
    whfast->tree_N = r->N;
    const size_t node_count = reb_integrator_whfast_hj_node_count(r->N);
    if (node_count > 0){
        whfast->nodes = calloc(node_count, sizeof(*whfast->nodes));
        if (whfast->nodes == NULL){
            reb_simulation_error(r, "WHFast HJ was not able to allocate memory for the fixed hierarchy.");
            return 1;
        }
    }
    return 0;
}

REB_API int reb_integrator_whfast_hj_set_tree(struct reb_simulation* const r, const char* const tree)
{
    if (r == NULL || tree == NULL){
        return 1;
    }
    if (r->integrator.name == NULL || strcmp(r->integrator.name, "whfast_hj") != 0 || r->integrator.state == NULL){
        reb_simulation_error(r, "WHFast HJ tree can only be set when the selected integrator is whfast_hj.");
        return 1;
    }

    struct reb_integrator_whfast_hj_state candidate = {0};
    if (reb_integrator_whfast_hj_allocate_fixed_tree(r, &candidate)){
        return 1;
    }

    unsigned char* const used = r->N > 0 ? calloc(r->N, sizeof(*used)) : NULL;
    if (r->N > 0 && used == NULL){
        reb_integrator_whfast_hj_clear_state(&candidate);
        reb_simulation_error(r, "WHFast HJ was not able to allocate memory for tree validation.");
        return 1;
    }

    struct reb_integrator_whfast_hj_tree_parser parser = {
        .r = r,
        .whfast = &candidate,
        .cursor = tree,
        .used = used,
        .next_internal = r->N,
    };
    const size_t root = reb_integrator_whfast_hj_parse_tree_node(&parser);
    if (parser.error == NULL){
        reb_integrator_whfast_hj_tree_parser_skip_space(&parser);
        if (*parser.cursor != '\0'){
            reb_integrator_whfast_hj_tree_parser_error(&parser, "Invalid WHFast HJ tree: trailing characters.");
        }else if (root == SIZE_MAX && r->N != 0){
            reb_integrator_whfast_hj_tree_parser_error(&parser, "Invalid WHFast HJ tree: empty tree for a non-empty simulation.");
        }else if (parser.leaf_count != r->N){
            reb_integrator_whfast_hj_tree_parser_error(&parser, "Invalid WHFast HJ tree: tree must include every particle exactly once.");
        }else if (parser.next_internal != reb_integrator_whfast_hj_node_count(r->N)){
            reb_integrator_whfast_hj_tree_parser_error(&parser, "Invalid WHFast HJ tree: expected a full binary hierarchy.");
        }
    }
    free(used);

    if (parser.error != NULL){
        reb_integrator_whfast_hj_clear_state(&candidate);
        reb_simulation_error(r, parser.error);
        return 1;
    }

    candidate.given_tree = 1;
    struct reb_integrator_whfast_hj_state* const whfast = r->integrator.state;
    reb_integrator_whfast_hj_clear_state(whfast);
    *whfast = candidate;
    return 0;
}

REB_API int reb_integrator_whfast_hj_set_binary_plus_particles_tree(struct reb_simulation* const r)
{
    if (r == NULL){
        return 1;
    }
    if (r->integrator.name == NULL || strcmp(r->integrator.name, "whfast_hj") != 0 || r->integrator.state == NULL){
        reb_simulation_error(r, "WHFast HJ tree can only be set when the selected integrator is whfast_hj.");
        return 1;
    }

    struct reb_integrator_whfast_hj_state candidate = {0};
    if (reb_integrator_whfast_hj_allocate_fixed_tree(r, &candidate)){
        return 1;
    }

    for (size_t i=0; i<r->N; i++){
        reb_integrator_whfast_hj_initialize_leaf(&candidate.nodes[i], r->particles[i].m);
    }
    if (r->N > 0){
        size_t root = 0;
        for (size_t i=1; i<r->N; i++){
            const size_t index = r->N + i - 1;
            if (reb_integrator_whfast_hj_initialize_internal(&candidate, index, root, i)){
                reb_integrator_whfast_hj_clear_state(&candidate);
                reb_simulation_error(r, "WHFast HJ requires each binary orbit to have positive total mass.");
                return 1;
            }
            root = index;
        }
    }

    candidate.given_tree = 1;
    struct reb_integrator_whfast_hj_state* const whfast = r->integrator.state;
    reb_integrator_whfast_hj_clear_state(whfast);
    *whfast = candidate;
    return 0;
}

REB_API void reb_integrator_whfast_hj_clear_tree(struct reb_simulation* const r)
{
    if (r == NULL || r->integrator.name == NULL || strcmp(r->integrator.name, "whfast_hj") != 0 || r->integrator.state == NULL){
        return;
    }
    reb_integrator_whfast_hj_clear_state(r->integrator.state);
}

static void reb_integrator_whfast_hj_tree_append(
    char* const buffer,
    const size_t buffer_size,
    size_t* const required,
    const char* const text
){
    const size_t len = strlen(text);
    if (buffer != NULL && buffer_size > 0 && *required < buffer_size - 1){
        size_t copy_len = buffer_size - 1 - *required;
        if (copy_len > len){
            copy_len = len;
        }
        memcpy(buffer + *required, text, copy_len);
        buffer[*required + copy_len] = '\0';
    }
    *required += len;
}

static void reb_integrator_whfast_hj_tree_node_to_string(
    const struct reb_integrator_whfast_hj_state* const whfast,
    const size_t index,
    char* const buffer,
    const size_t buffer_size,
    size_t* const required
){
    const struct reb_integrator_whfast_hj_node* const node = &whfast->nodes[index];
    if (index < whfast->tree_N){
        char leaf[32];
        snprintf(leaf, sizeof(leaf), "%zu", index + 1);
        reb_integrator_whfast_hj_tree_append(buffer, buffer_size, required, leaf);
        return;
    }
    reb_integrator_whfast_hj_tree_append(buffer, buffer_size, required, "[");
    reb_integrator_whfast_hj_tree_node_to_string(whfast, node->primary, buffer, buffer_size, required);
    reb_integrator_whfast_hj_tree_append(buffer, buffer_size, required, ",");
    reb_integrator_whfast_hj_tree_node_to_string(whfast, node->secondary, buffer, buffer_size, required);
    reb_integrator_whfast_hj_tree_append(buffer, buffer_size, required, "]");
}

REB_API int reb_integrator_whfast_hj_tree_to_string(
    struct reb_simulation* const r,
    char* const buffer,
    const size_t buffer_size
){
    if (buffer != NULL && buffer_size > 0){
        buffer[0] = '\0';
    }
    if (r == NULL || r->integrator.name == NULL || strcmp(r->integrator.name, "whfast_hj") != 0 || r->integrator.state == NULL){
        return -1;
    }
    const struct reb_integrator_whfast_hj_state* const whfast = r->integrator.state;
    if (!whfast->given_tree || (whfast->tree_N > 0 && whfast->nodes == NULL)){
        return -1;
    }

    size_t required = 0;
    const size_t node_count = reb_integrator_whfast_hj_node_count(whfast->tree_N);
    if (node_count == 0){
        reb_integrator_whfast_hj_tree_append(buffer, buffer_size, &required, "[]");
    }else{
        reb_integrator_whfast_hj_tree_node_to_string(whfast, node_count - 1, buffer, buffer_size, &required);
    }
    return required > (size_t)INT_MAX ? -1 : (int)required;
}

static void reb_integrator_whfast_hj_from_inertial(
    struct reb_simulation* const r,
    struct reb_integrator_whfast_hj_state* const whfast
){
    for (size_t i=0; i<whfast->tree_N; i++){
        whfast->nodes[i].barycenter_particle = r->particles[i];
    }
    const size_t node_count = reb_integrator_whfast_hj_node_count(whfast->tree_N);
    for (size_t i=whfast->tree_N; i<node_count; i++){
        reb_integrator_whfast_hj_update_internal_coordinate(whfast, i);
    }
}

static void reb_integrator_whfast_hj_reconstruct_children(
    struct reb_integrator_whfast_hj_state* const whfast,
    const size_t index
){
    const struct reb_integrator_whfast_hj_node* const node = &whfast->nodes[index];
    const struct reb_particle barycenter = node->barycenter_particle;
    const struct reb_particle jacobi = node->jacobi_particle;
    struct reb_particle* const primary = &whfast->nodes[node->primary].barycenter_particle;
    struct reb_particle* const secondary = &whfast->nodes[node->secondary].barycenter_particle;

    primary->x = barycenter.x - node->primary_offset*jacobi.x;
    primary->y = barycenter.y - node->primary_offset*jacobi.y;
    primary->z = barycenter.z - node->primary_offset*jacobi.z;
    primary->vx = barycenter.vx - node->primary_offset*jacobi.vx;
    primary->vy = barycenter.vy - node->primary_offset*jacobi.vy;
    primary->vz = barycenter.vz - node->primary_offset*jacobi.vz;
    primary->ax = barycenter.ax - node->primary_offset*jacobi.ax;
    primary->ay = barycenter.ay - node->primary_offset*jacobi.ay;
    primary->az = barycenter.az - node->primary_offset*jacobi.az;

    secondary->x = barycenter.x + node->secondary_offset*jacobi.x;
    secondary->y = barycenter.y + node->secondary_offset*jacobi.y;
    secondary->z = barycenter.z + node->secondary_offset*jacobi.z;
    secondary->vx = barycenter.vx + node->secondary_offset*jacobi.vx;
    secondary->vy = barycenter.vy + node->secondary_offset*jacobi.vy;
    secondary->vz = barycenter.vz + node->secondary_offset*jacobi.vz;
    secondary->ax = barycenter.ax + node->secondary_offset*jacobi.ax;
    secondary->ay = barycenter.ay + node->secondary_offset*jacobi.ay;
    secondary->az = barycenter.az + node->secondary_offset*jacobi.az;
}

static void reb_integrator_whfast_hj_to_inertial(
    struct reb_simulation* const r,
    struct reb_integrator_whfast_hj_state* const whfast
){
    const size_t node_count = reb_integrator_whfast_hj_node_count(whfast->tree_N);
    for (size_t i=node_count; i-- > whfast->tree_N;){
        reb_integrator_whfast_hj_reconstruct_children(whfast, i);
    }
    for (size_t i=0; i<whfast->tree_N; i++){
        const struct reb_particle source = whfast->nodes[i].barycenter_particle;
        struct reb_particle* const particle = &r->particles[i];
        particle->x = source.x;
        particle->y = source.y;
        particle->z = source.z;
        particle->vx = source.vx;
        particle->vy = source.vy;
        particle->vz = source.vz;
    }
}

static void reb_integrator_whfast_hj_interaction_step(
    const struct reb_simulation* const r,
    struct reb_integrator_whfast_hj_state* const whfast,
    const double dt
){
    const size_t node_count = reb_integrator_whfast_hj_node_count(whfast->tree_N);
    for (size_t i=whfast->tree_N; i<node_count; i++){
        struct reb_integrator_whfast_hj_node* const node = &whfast->nodes[i];
        struct reb_particle* const p = &node->jacobi_particle;
        p->vx += dt*p->ax;
        p->vy += dt*p->ay;
        p->vz += dt*p->az;

        const double inverse_r2 = 1./(p->x*p->x + p->y*p->y + p->z*p->z);
        const double inverse_r = sqrt(inverse_r2);
        const double prefactor = dt*r->G*node->barycenter_particle.m*inverse_r*inverse_r2;
        p->vx += prefactor*p->x;
        p->vy += prefactor*p->y;
        p->vz += prefactor*p->z;
    }
}

static void reb_integrator_whfast_hj_kepler_step(
    const struct reb_simulation* const r,
    struct reb_integrator_whfast_hj_state* const whfast,
    const double dt
){
    const size_t node_count = reb_integrator_whfast_hj_node_count(whfast->tree_N);
    for (size_t i=whfast->tree_N; i<node_count; i++){
        struct reb_integrator_whfast_hj_node* const node = &whfast->nodes[i];
        reb_integrator_whfast_kepler_solver(&node->jacobi_particle, node->barycenter_particle.m*r->G, dt, r);
    }
}

static void reb_integrator_whfast_hj_com_step(
    struct reb_integrator_whfast_hj_state* const whfast,
    const double dt
){
    const size_t node_count = reb_integrator_whfast_hj_node_count(whfast->tree_N);
    if (node_count == 0){
        return;
    }
    struct reb_particle* const com = &whfast->nodes[node_count - 1].barycenter_particle;
    com->x += dt*com->vx;
    com->y += dt*com->vy;
    com->z += dt*com->vz;
}

void reb_integrator_whfast_hj_step(struct reb_simulation* const r, void* state)
{
    struct reb_integrator_whfast_hj_state* const whfast = state;
    const double dt = r->dt;

    if (!whfast->given_tree || (r->N > 0 && whfast->nodes == NULL)){
        reb_simulation_error(r, "WHFast HJ requires a fixed tree. Provide one with integrate(..., given_tree=True, tree=...).");
        return;
    }
    if (whfast->tree_N != r->N){
        reb_simulation_error(r, "WHFast HJ particle count changed after the fixed tree was set.");
        return;
    }
    for (size_t i=0; i<whfast->tree_N; i++){
        if (r->particles[i].m != whfast->nodes[i].barycenter_particle.m){
            reb_simulation_error(r, "WHFast HJ requires fixed particle masses after the hierarchy is set.");
            return;
        }
    }
    reb_integrator_whfast_hj_from_inertial(r, whfast);

    reb_integrator_whfast_hj_kepler_step(r, whfast, dt/2.);
    reb_integrator_whfast_hj_com_step(whfast, dt/2.);
    reb_integrator_whfast_hj_to_inertial(r, whfast);

    r->gravity_ignore_terms = REB_GRAVITY_IGNORE_TERMS_NONE;
    reb_simulation_update_acceleration(r);
    reb_integrator_whfast_hj_from_inertial(r, whfast);
    reb_integrator_whfast_hj_interaction_step(r, whfast, dt);

    reb_integrator_whfast_hj_kepler_step(r, whfast, dt/2.);
    reb_integrator_whfast_hj_com_step(whfast, dt/2.);
    reb_integrator_whfast_hj_to_inertial(r, whfast);

    r->t += dt;
    r->dt_last_done = dt;
}
