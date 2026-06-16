#include "F_surgical_policy.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <Python.h>

// ANSI color codes for terminal output
#define COLOR_RED     "\033[1;31m"
#define COLOR_GREEN   "\033[1;32m"
#define COLOR_CYAN    "\033[1;36m"
#define COLOR_YELLOW  "\033[1;33m"
#define COLOR_BOLD    "\033[1m"
#define COLOR_RESET   "\033[0m"

// Separator line width
#define SEP "═══════════════════════════════════════════════════════════════"

// Tool class names matching the detection model's 6-class output
static const char* TOOL_NAMES[6] = {
    "Background", "Suction", "Bipolar Forceps", "Other", "Dissecting Forceps", "Cusa"
};

// Print a clean, colored per-frame status report
// Uses RAW inputs (before enforcer modification) to detect violations
void print_frame_report(int frame, inputs_surgical_policy_t* raw_in, int presence[6]) {
    printf("\n" COLOR_CYAN SEP COLOR_RESET "\n");
    printf(COLOR_BOLD "  FRAME %02d" COLOR_RESET "\n", frame);
    printf(COLOR_CYAN "───────────────────────────────────────────────────────────────" COLOR_RESET "\n");
    
    // Print ALL tools detected in this frame (from full presence vector)
    printf(COLOR_BOLD "  Tools Present in Frame: " COLOR_RESET);
    int any_tool = 0;
    for (int i = 0; i < 6; i++) {
        if (presence[i]) {
            if (any_tool) printf(", ");
            printf("%s", TOOL_NAMES[i]);
            any_tool = 1;
        }
    }
    if (!any_tool) printf("(none)");
    printf("\n");

    // Print consecutive ON counts
    printf(COLOR_BOLD "  Consecutive ON:  " COLOR_RESET
           "Suction=%u  Bipolar=%u  Dissecting=%u  Cusa=%u\n",
           raw_in->CONSEC_Suction, raw_in->CONSEC_Bipolar, 
           raw_in->CONSEC_Dissecting, raw_in->CONSEC_Cusa);

    // Print consecutive OFF counts
    printf(COLOR_BOLD "  Consecutive OFF: " COLOR_RESET
           "Suction=%u  Bipolar=%u  Dissecting=%u  Cusa=%u\n",
           raw_in->CONSEC_OFF_Suction, raw_in->CONSEC_OFF_Bipolar,
           raw_in->CONSEC_OFF_Dissecting, raw_in->CONSEC_OFF_Cusa);

    // Print cumulative and flags
    printf(COLOR_BOLD "  Total Suction: " COLOR_RESET "%u" 
           COLOR_BOLD "  | Cooldown Dissecting: " COLOR_RESET "%s"
           COLOR_BOLD "  | Dissecting Ever Used: " COLOR_RESET "%s\n",
           raw_in->TOTAL_Suction,
           raw_in->IN_COOLDOWN_Dissecting ? "YES" : "NO",
           raw_in->Dissecting_Ever_Used ? "YES" : "NO");

    printf(COLOR_CYAN "───────────────────────────────────────────────────────────────" COLOR_RESET "\n");
    printf(COLOR_BOLD "  Policy Evaluation Results:\n" COLOR_RESET);

    // Check violations directly from RAW inputs (before enforcer modifies them)

    // --- Policy P1: Suction + Bipolar conflict ---
    if (raw_in->Suction && raw_in->Bipolar) {
        printf(COLOR_RED "    ✗ P1 VIOLATED: Suction and Bipolar are active together!" COLOR_RESET "\n");
    } else {
        printf(COLOR_GREEN "    ✓ P1 OK: No Suction + Bipolar conflict." COLOR_RESET "\n");
    }

    // --- Policy P2: Dissecting max continuous usage ---
    if (raw_in->Dissecting && raw_in->CONSEC_Dissecting > CONST_p2_dissecting_time_MAX_Dissecting) {
        printf(COLOR_RED "    ✗ P2 VIOLATED: Dissecting exceeded max continuous usage (%d frames)!" COLOR_RESET "\n",
               CONST_p2_dissecting_time_MAX_Dissecting);
    } else {
        printf(COLOR_GREEN "    ✓ P2 OK: Dissecting continuous usage within limits." COLOR_RESET "\n");
    }

    // --- Policy P3: Suction cumulative total usage ---
    if (raw_in->Suction && raw_in->TOTAL_Suction > CONST_p3_total_suction_MAX_TOTAL_Suction) {
        printf(COLOR_RED "    ✗ P3 VIOLATED: Suction total usage exceeded %d frames!" COLOR_RESET "\n",
               CONST_p3_total_suction_MAX_TOTAL_Suction);
    } else {
        printf(COLOR_GREEN "    ✓ P3 OK: Suction total usage within limits." COLOR_RESET "\n");
    }

    // --- Policy P4: Dissecting cooldown ---
    if (raw_in->Dissecting && raw_in->IN_COOLDOWN_Dissecting) {
        printf(COLOR_RED "    ✗ P4 VIOLATED: Dissecting is active during cooldown period!" COLOR_RESET "\n");
    } else {
        printf(COLOR_GREEN "    ✓ P4 OK: Dissecting cooldown satisfied." COLOR_RESET "\n");
    }

    // --- Policy P5: Cusa requires prior Dissecting usage ---
    if (raw_in->Cusa && !raw_in->Dissecting_Ever_Used) {
        printf(COLOR_RED "    ✗ P5 VIOLATED: Cusa used without prior Dissecting usage!" COLOR_RESET "\n");
    } else {
        printf(COLOR_GREEN "    ✓ P5 OK: Cusa precedence satisfied." COLOR_RESET "\n");
    }

    printf(COLOR_CYAN SEP COLOR_RESET "\n");
}

int main() {
    enforcervars_surgical_policy_t enf;
    inputs_surgical_policy_t inputs;
    outputs_surgical_policy_t outputs;

    // Initial setup
    surgical_policy_init_all_vars(&enf, &inputs, &outputs);

    printf(COLOR_BOLD "\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║     SURGICAL POLICY ENFORCER — RUNTIME MONITOR OUTPUT       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n" COLOR_RESET);

    // ---------------- PYTHON INIT ----------------
    Py_Initialize();

    PyRun_SimpleString(
        "import sys; import os; "
        "sys.path.append('.'); "
        "sys.argv = [''];"
    );

    PyObject *module = PyImport_ImportModule("test_image");
    if (!module) {
        PyErr_Print();
        return 1;
    }

    // ---------------- MAIN LOOP ----------------
    for (int frame = 0; frame < 20; frame++) {

        // ===== RESET FSM states to s0 for independent per-frame evaluation =====
        enf._policy_p1_state = POLICY_STATE_surgical_policy_p1_s0;
        enf._policy_p2_dissecting_time_state = POLICY_STATE_surgical_policy_p2_dissecting_time_s0;
        enf._policy_p3_total_suction_state = POLICY_STATE_surgical_policy_p3_total_suction_s0;
        enf._policy_p4_cooldown_dissecting_state = POLICY_STATE_surgical_policy_p4_cooldown_dissecting_s0;
        enf._policy_p5_precedence_cusa_state = POLICY_STATE_surgical_policy_p5_precedence_cusa_s0;

        // Reset all inputs
        inputs.Suction = 0;
        inputs.Bipolar = 0;
        inputs.Dissecting = 0;
        inputs.Cusa = 0;
        inputs.CONSEC_Suction = 0;
        inputs.CONSEC_Bipolar = 0;
        inputs.CONSEC_Dissecting = 0;
        inputs.CONSEC_Cusa = 0;
        inputs.CONSEC_OFF_Suction = 0;
        inputs.CONSEC_OFF_Bipolar = 0;
        inputs.CONSEC_OFF_Dissecting = 0;
        inputs.CONSEC_OFF_Cusa = 0;
        inputs.TOTAL_Suction = 0;
        inputs.IN_COOLDOWN_Dissecting = 0;
        inputs.Dissecting_Ever_Used = 0;

        PyObject *result = PyObject_CallMethod(module, "test_frame", "i", frame);
        if (!result) {
            PyErr_Print();
            break;
        }

        // Full presence vector for all 6 tool classes
        int presence[6] = {0};

        if (PyList_Check(result)) {
            int len = PyList_Size(result);
            int arr[20] = {0};

            for (int i = 0; i < len && i < 20; i++) {
                arr[i] = PyLong_AsLong(PyList_GetItem(result, i));
            }
            // -------- MAPPING --------
            // index 0  → Background
            // index 1  → Suction
            // index 2  → Bipolar Forceps
            // index 3  → Other
            // index 4  → Dissecting Forceps
            // index 5  → Cusa
            // index 6  → consec_suction
            // index 7  → consec_bipolar
            // index 8  → consec_dissecting
            // index 9  → consec_cusa
            // index 10 → consec_off_suction
            // index 11 → consec_off_bipolar
            // index 12 → consec_off_dissecting
            // index 13 → consec_off_cusa
            // index 14 → total_suction
            // index 15 → in_cooldown_dissecting
            // index 16 → dissecting_ever_used

            // Capture full presence vector (all 6 tool classes)
            for (int i = 0; i < 6; i++) {
                presence[i] = arr[i];
            }

            // Tool presence booleans
            inputs.Suction    = arr[1];
            inputs.Bipolar    = arr[2];
            inputs.Dissecting = arr[4];  // skip index 3 (Other)
            inputs.Cusa       = arr[5];

            // Consecutive ON counts
            inputs.CONSEC_Suction    = arr[6];
            inputs.CONSEC_Bipolar    = arr[7];
            inputs.CONSEC_Dissecting = arr[8];
            inputs.CONSEC_Cusa       = arr[9];

            // Consecutive OFF counts
            inputs.CONSEC_OFF_Suction    = arr[10];
            inputs.CONSEC_OFF_Bipolar    = arr[11];
            inputs.CONSEC_OFF_Dissecting = arr[12];
            inputs.CONSEC_OFF_Cusa       = arr[13];

            // Cumulative and flag inputs
            inputs.TOTAL_Suction         = arr[14];
            inputs.IN_COOLDOWN_Dissecting = arr[15];
            inputs.Dissecting_Ever_Used  = arr[16];
        }

        Py_CLEAR(result);

        // Save raw inputs BEFORE the enforcer modifies them
        inputs_surgical_policy_t raw_inputs = inputs;

        // Run enforcer (each frame starts from s0 — independent evaluation)
        surgical_policy_run_via_enforcer(&enf, &inputs, &outputs);

        // Print colored per-frame report using RAW inputs for violation detection
        print_frame_report(frame + 1, &raw_inputs, presence);
    }

    // ---------------- CLEANUP ----------------
    Py_CLEAR(module);
    Py_Finalize();

    printf(COLOR_BOLD "\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                    SIMULATION COMPLETE                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n" COLOR_RESET);
    return 0;
}

// Required dummy controller — no-op (policies are purely monitoring)
void surgical_policy_run(inputs_surgical_policy_t *inputs, outputs_surgical_policy_t *outputs) {
    // no-op
}