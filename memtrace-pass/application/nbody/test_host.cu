//
// N-BODY CODE
//

#include <iostream>
#include <sys/stat.h> // for file existence check
#include <random>
#include <cuda.h>
#include <chrono>
#include <string>
#include <vector>
#include <math.h>
#include <stdexcept>
#include <iomanip>
#include <fstream>
#include <new> // std::bad_alloc //
#include <algorithm> // std::swap //

using namespace std;
using Clock = chrono::high_resolution_clock;
using Duration = chrono::duration<double>;

// precision for file output
#define FILEOUTPUT_PRECISION 10

__global__
void up_kernel(float* pos_x,
                     float* pos_y,
                     float* pos_z,
                     const float* vel_x,
                     const float* vel_y,
                     const float* vel_z,
                     float dt,
                     int N) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    pos_x[id] += vel_x[id] * dt;
    pos_y[id] += vel_y[id] * dt;
    pos_z[id] += vel_z[id] * dt;
}

__global__
void us_kernel(const float* masses,
                 const float* pos_x,
                 const float* pos_y,
                 const float* pos_z,
                 float* vel_x,
                 float* vel_y,
                 float* vel_z,
                 float dt,
                 float epsilon,
                 int N) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // the body's acceleration
    float acc_x = 0;
    float acc_y = 0;
    float acc_z = 0;

    // temporary register
    float diff_x;
    float diff_y;
    float diff_z;
    float norm;
    int j;

    for (j = 0; j < N; ++j) {
        diff_x = pos_x[j] - pos_x[id];
        diff_y = pos_y[j] - pos_y[id];
        diff_z = pos_z[j] - pos_z[id];

        // to ensure a certain order of execution we write
        // the calculations in seperate lines. Keep in mind
        // that opencl does not define an operator precedence,
        // thus we have to ensure this by ourselves.
        norm = diff_x * diff_x;
        norm += diff_y * diff_y;
        norm += diff_z * diff_z;
        norm = sqrt(norm);
        norm = norm * norm * norm;
        norm = norm == 0 ? 0 : 1.0f / norm + epsilon;
        norm *= masses[j];

        acc_x += norm * diff_x;
        acc_y += norm * diff_y;
        acc_z += norm * diff_z;
    }

    vel_x[id] += acc_x * dt;
    vel_y[id] += acc_y * dt;
    vel_z[id] += acc_z * dt;
}

template<class IntType>
void commandLineGetInt(IntType* p, const std::string& key, int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == key) {
            *p = std::stoll(argv[i+1]);
            break;
        }
    }
}

void commandLineGetFloat(float* p, const std::string& key, int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == key) {
            *p = std::stof(argv[i+1]);
            break;
        }
    }
}

void commandLineGetString(string* s, const std::string& key, int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == key) {
            s->assign(argv[i + 1]);
            break;
        }
    }
}

void commandLineGetBool(bool* p, const std::string& key, int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == key) {
            *p = true;
            break;
        }
    }
}

bool commandLineGetBool(const std::string& key, int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == key) {
            return true;
        }
    }
    return false;
}

bool fileExists(const std::string& filename) {
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1) {
        return true;
    }
    return false;
}

// gives back the relative number of different elements,
// thus the result will be between 0 and 1.
template<class ArrayA_T, class ArrayB_T>
double relNumDiffEl(ArrayA_T a, ArrayB_T b, int N, double diff = 0) {
    double res = 0;
    for (int i = 0; i < N; ++i) {
        if (fabs(a[i] - b[i]) > diff) {
            res += 1. / (float) N;
        }
    }
    return res;
}

// gives back the mean of the relative differences
// between the elements.
template<class ArrayA_T, class ArrayB_T>
double relDiffEl(ArrayA_T a, ArrayB_T b, int N) {
    double res = 0;
    for (int i = 0; i < N; ++i) {
        res += fabs(a[i] - b[i])/b[i];
    }
    return res;
}

void updatePositions_cpu(float* pos_x,
                         float* pos_y,
                         float* pos_z,
                         const float* vel_x,
                         const float* vel_y,
                         const float* vel_z,
                         float dt,
                         int N) {

    for (int id = 0; id < N; ++id) {
        pos_x[id] = pos_x[id] + vel_x[id] * dt;
        pos_y[id] = pos_y[id] + vel_y[id] * dt;
        pos_z[id] = pos_z[id] + vel_z[id] * dt;
    }
}

void updateSpeed_cpu(const float* masses,
                     const float* pos_x,
                     const float* pos_y,
                     const float* pos_z,
                     float* vel_x,
                     float* vel_y,
                     float* vel_z,
                     float dt,
                     float epsilon,
                     int N) {

    // the body's acceleration
    float acc_x;
    float acc_y;
    float acc_z;

    // temporary register
    float diff_x;
    float diff_y;
    float diff_z;
    float norm;
    int j;

    for (int id = 0; id < N; ++id) {
        acc_x = 0;
        acc_y = 0;
        acc_z = 0;

        for (j = 0; j < N; ++j) {
            diff_x = pos_x[j] - pos_x[id];
            diff_y = pos_y[j] - pos_y[id];
            diff_z = pos_z[j] - pos_z[id];

            // to ensure a certain order of execution we write
            // the calculations in seperate lines. Keep in mind
            // that opencl does not define an operator precedence,
            // thus we have to ensure this by ourselves.
            norm = diff_x * diff_x;
            norm += diff_y * diff_y;
            norm += diff_z * diff_z;
            norm = sqrt(norm);
            norm = norm * norm * norm;
            norm = norm == 0 ? 0 : 1.0f / norm + epsilon;
            norm *= masses[j];

            acc_x += norm * diff_x;
            acc_y += norm * diff_y;
            acc_z += norm * diff_z;
        }

        vel_x[id] += acc_x * dt;
        vel_y[id] += acc_y * dt;
        vel_z[id] += acc_z * dt;
    }
} // end updateSpeed_cpu()

template<class ArrayT>
void printArray(ArrayT p, int N, int precision = 4) {
    for (int i = 0; i < N - 1; ++i) {
        cout << setprecision(precision) << setw(precision + 4) << left << p[i] << ' ';
    }
    cout << p[N - 1];
}

void printUsage(const char* fileName) {
    cout << endl;
    cout << "Usage: " << fileName << " [OPTION]"<< endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << "  -N <size>" << endl;
    cout << "     denotes the number of bodies to simulate" << endl;
    cout << "  -T <steps>" << endl;
    cout << "     number of time steps to integrate the motion" << endl;
    cout << "  -dt <time step>" << endl;
    cout << "     size of one timestep (default = 0.001)" << endl;
    cout << "  -e <softening>" << endl;
    cout << "     gravitational softening to smooth the bodies' motion (default = 0.001)" << endl;
    cout << "  -bs <block size>" << endl;
    cout << "     specifies the number of threads per block" << endl;
    cout << "  -check" << endl;
    cout << "     progam verifies the results with the cpu calculations" << endl;
    cout << "     and report the error" << endl;
    cout << "  -v" << endl;
    cout << "     print the bodies' properties after `T` timesteps for gpu (and cpu)" << endl;
    cout << "  -fno-pu" << endl;
    cout << "     force no updates of the particles' positions" << endl;
    cout << "  -fno-su" << endl;
    cout << "     force no updates of the particles' velocities" << endl;
    cout << "  -fout <output file>" << endl;
    cout << "     e.g. ./results/out.txt the programm will create" << endl;
    cout << "     a file containing the bodies' properties for every timestep." << endl;
    cout << "     If available the CPU output will be written." << endl;
    cout << "  --write-only-last" << endl;
    cout << "     If set only the last timestep will be written to the specified" << endl;
    cout << "     output file" << endl;
    cout << "  -fkernel <.ptx file>" << endl;
    cout << "     the file with kernels `updatePositions` and `updateSpeed`" << endl;
    cout << "     (default = ./n-body-kernel.ptx)" << endl;
    cout << "  -fcheck-file <.csv file>" << endl;
    cout << "     Checks the current result against the result stored in the given" << endl;
    cout << "     file. The file must contain 7 lines with the positions in x y z," << endl;
    cout << "     the velocities in x y z and the masses. Every line must contain" << endl;
    cout << "     the same number of values seperated by whitespaces." << endl;
    cout << "  -h" << endl;
    cout << "     show this help message" << endl;
    cout << endl;
}

int main(int argc, char* argv[]) {

    // Default Values //
    int N = 1024;
    int T = 1000;
    float dt = 0.0001f;
    float e = 0.01f;
    int blockSize = 64;
    bool checkResult = false;
    bool writeOnlyLast = false;
    bool verbose = false;
    bool fnopu = false; // force no position update
    bool fnosu = false; // force no speed update

    string fout;
    string fcheckfile;
    string fkernel = "n-body-kernel.ptx";

    if (commandLineGetBool("-h", argc, argv) || commandLineGetBool("--help", argc, argv)) {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    commandLineGetInt(&N, "-N", argc, argv);
    commandLineGetInt(&T, "-T", argc, argv);
    commandLineGetFloat(&dt, "-dt", argc, argv);
    commandLineGetFloat(&e, "-e", argc, argv);
    commandLineGetInt(&blockSize, "-bs", argc, argv);
    checkResult = commandLineGetBool("-check", argc, argv);
    verbose = commandLineGetBool("-v", argc, argv);
    fnopu = commandLineGetBool("-fno-pu", argc, argv);
    fnosu = commandLineGetBool("-fno-su", argc, argv);
    commandLineGetString(&fout, "-fout", argc, argv);
    commandLineGetString(&fcheckfile, "-fcheck-file", argc, argv);
    commandLineGetString(&fkernel, "-fkernel", argc, argv);
    writeOnlyLast = commandLineGetBool("--write-only-last", argc, argv);

    if (!fout.empty()) {
        if (fileExists(fout)) {
            cout << "***WARNING: file '" << fout << "' already exists; ";
            cout << "Output will be appended!" << endl;
        }
    }

    unsigned pd = 35; // padding for output
    cout << left << boolalpha;
    cout << endl;
    cout << "# N-Body Computation" << endl;
    cout << endl;
    cout << "## Input Arguments" << endl;
    cout << endl;
    cout << "  - N  = " << setw(pd) << N << "(N bodies in total)" << endl;
    cout << "  - T  = " << setw(pd) << T << "(timesteps in total)" << endl;
    cout << "  - dt = " << setw(pd) << dt << "(size of one timesteps)" << endl;
    cout << "  - e  = " << setw(pd) << e << "(gravitational softening)" << endl;
    cout << "  - bs = " << setw(pd) << blockSize << "(number of threads per block)" << endl;
    cout << "  - checkResult = " << setw(pd-9) << checkResult << "(execute cpu n-body verification)" << endl;
    cout << "  - write-only-last = " << setw(pd-9) << writeOnlyLast << "(execute cpu n-body verification)" << endl;
    cout << "  - verbose     = " << setw(pd-9) << verbose << endl;
    cout << "  - fnopu       = " << setw(pd-9) << fnopu << endl;
    cout << "  - fnosu       = " << setw(pd-9) << fnosu << endl;
    cout << "  - fout        = " << setw(pd-9) << (fout.empty() ? "<not specified>" : fout) << "(output file)" << endl;
    cout << "  - fcheck-file = " << setw(pd-9) << (fcheckfile.empty() ? "<not specified>" : fcheckfile) << "(external result)" << endl;
    cout << "  - fkernel     = " << setw(pd-9) << fkernel << "(.ptx kernel file)" << endl;
    cout << endl;
    cout << "## Execution Progress" << endl;
    cout << endl;

    // ALLOCATE HOST MEMORY //
    cout << "  - Going to allocate host memory..." << flush;
    float* pos_x_h; // positions
    float* pos_y_h;
    float* pos_z_h;
    float* vel_x_h; // velocities
    float* vel_y_h;
    float* vel_z_h; 
    float* m_h;     // masses
    float** all_h[] = { &pos_x_h, &pos_y_h, &pos_z_h, &vel_x_h, &vel_y_h, &vel_z_h, &m_h };
    try {
        for (int i = 0; i < 7; ++i) {
            *all_h[i] = new float[N];
        }
    } catch (bad_alloc& ba) {
        cout << "[FAILED]" << endl;
        return EXIT_FAILURE;
    }
    cout << "[OK]" << endl;

    // IF WE WANT TO COMPARE GPU VS CPU RESULT WE NEED ADDITIONAL BUFFERS
    float* pos_x_gpures; // positions
    float* pos_y_gpures;
    float* pos_z_gpures;
    float* vel_x_gpures; // velocities
    float* vel_y_gpures;
    float* vel_z_gpures; 
    float* m_gpures;     // masses
    float** all_gpures[] = { &pos_x_gpures, &pos_y_gpures, &pos_z_gpures,
                             &vel_x_gpures, &vel_y_gpures, &vel_z_gpures, &m_gpures };
    if (checkResult) {
        cout << "  - Going to allocate host memory for GPU result check..." << flush;
        try {
            for (int i = 0; i < 7; ++i) {
                *all_gpures[i] = new float[N];
            }
        } catch (bad_alloc& ba) {
            cout << "[FAILED]" << endl;
            return EXIT_FAILURE;
        }
        cout << "[OK]" << endl;
    }

    // SET INITIAL CONDITIONS //
    cout << "  - Going to set initial conditions..." << flush;
    default_random_engine gen(0);
    uniform_real_distribution<float> dist(0, 1);
    for (unsigned int i = 0; i < N; ++i) {
        vel_x_h[i] = 0;
        vel_y_h[i] = 0;
        vel_z_h[i] = 0;
        pos_x_h[i] = dist(gen);
        pos_y_h[i] = dist(gen);
        pos_z_h[i] = dist(gen);
        m_h[i] = dist(gen) + 1;
    }
    cout << "[OK]" << endl;

    // ALLOCATE DEVICE MEMORY //
    cout << "  - Allocating device memory..." << flush;
    size_t size = N * sizeof(float);
    float* pos_x_d;
    float* pos_y_d;
    float* pos_z_d;
    float* vel_x_d;
    float* vel_y_d;
    float* vel_z_d;
    float* m_d;
    float** all_d[] = { &pos_x_d, &pos_y_d, &pos_z_d, &vel_x_d, &vel_y_d, &vel_z_d, &m_d };
    for (int i = 0; i < 7; ++i) {
        if (cudaMalloc(all_d[i], size) != cudaSuccess) {
            cout << "[FAILED]" << endl;
            exit(1);
        }
    }
    cout << "[OK]" << endl;

    // COPY HOST TO DEVICE //
    cout << "  - Copy from host to device..." << flush;
    auto htod_begin = Clock::now();
    for (int i = 0; i < 7; ++i) {
        if (cudaMemcpy(*all_d[i], *all_h[i], size, cudaMemcpyHostToDevice) != cudaSuccess) {
            cout << "[FAILED]" << endl;
            exit(1);
        }
    }
    Duration t_htod = Clock::now() - htod_begin;
    cout << "[OK]" << endl;

    // VERIFY RESULTS WITH CPU CALCULATION //
    if (checkResult) {
        cout << "  - Execute CPU N-Body Calculation..." << flush;
        for (unsigned int i = 0; i < T; ++i) {
            if (!fout.empty() && !writeOnlyLast) {
                fstream fs (fout, fstream::out | fstream::app);
                for (int i = 0; i < 7; ++i) {
                    for (int j = 0; j < N - 1; ++j) {
                        fs << setprecision(FILEOUTPUT_PRECISION) << (*all_h[i])[j] << ' ';
                    }
                    fs << setprecision(FILEOUTPUT_PRECISION) << (*all_h[i])[N - 1] << endl;
                }
                fs.close();
            }
            updateSpeed_cpu(m_h, pos_x_h, pos_y_h, pos_z_h, vel_x_h, vel_y_h, vel_z_h, dt, e, N);
            updatePositions_cpu(pos_x_h, pos_y_h, pos_z_h, vel_x_h, vel_y_h, vel_z_h, dt, N);
        }
        cout << "[OK]" << endl;
    }

    // PREPARE KERNEL ARGUMENTS //
    int threads = blockSize;
    int blocks = N / blockSize;

    if ( N % blockSize != 0) {
        cout << "***ERROR: case N % blockSize != 0 is not supported" << endl
             << "***            now blockSize  = " << blockSize << endl;
        return EXIT_FAILURE;
    }
    
    int oc_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&oc_blocks, &us_kernel, threads, 0);
    cout << "  - Occupancy calculator result = " << oc_blocks << " active blocks per multiprocessor" << endl;

    cout << "  - Launching grids {" << blocks << ", 1, 1}, block {" << threads << ", 1, 1}..." << flush;
    auto kernel_begin = Clock::now();
    for (unsigned i = 0; i < T; ++i) {
        if (!fnosu) {
            us_kernel <<< blocks,threads >>> (m_d,pos_x_d,pos_y_d,pos_z_d,vel_x_d,vel_y_d,vel_z_d,dt,e,N);
            cudaDeviceSynchronize();
        }
        if (!fnopu) {
            up_kernel <<< blocks,threads >>> (pos_x_d,pos_y_d,pos_z_d,vel_x_d,vel_y_d,vel_z_d,dt,N);
            cudaDeviceSynchronize();
        }
        if (!fout.empty() && !checkResult && !writeOnlyLast) { // write gpu result to file
            for (int i = 0; i < 6; ++i) { // we skip copying the masses
                if (cudaMemcpy(*all_h[i], *all_d[i], size, cudaMemcpyDeviceToHost) != cudaSuccess) {
                    cout << "***ERROR while copying data from GPU to CPU" << endl;
                    exit(1);
                }
            }
            fstream fs (fout, fstream::out | fstream::app);
            for (int i = 0; i < 7; ++i) {
                for (int j = 0; j < N - 1; ++j) {
                    fs << setprecision(FILEOUTPUT_PRECISION) << (*all_h[i])[j] << ' ';
                }
                fs << setprecision(FILEOUTPUT_PRECISION) << (*all_h[i])[N - 1] << endl;
            }
            fs.close();
        }
    }
    Duration t_kernel = Clock::now() - kernel_begin;
    cout << "[OK]" << endl;

    // COPY DEVICE TO HOST //
    cout << "  - Copy from device to host..." << flush;
    auto dtoh_begin = Clock::now();
    if (checkResult) {
        for (int i = 0; i < 6; ++i) { // we skip copying the masses
            if (cudaMemcpy(*all_gpures[i], *all_d[i], size, cudaMemcpyDeviceToHost) != cudaSuccess) {
                cout << "[FAILED]" << endl;
                exit(1);
            }
        }
    }
    else {
        for (int i = 0; i < 6; ++i) { // we skip copying the masses
            if (cudaMemcpy(*all_h[i], *all_d[i], size, cudaMemcpyDeviceToHost) != cudaSuccess) {
                cout << "[FAILED]" << endl;
                exit(1);
            }
        }
    }
    Duration t_dtoh = Clock::now() - dtoh_begin;
    cout << "[OK]" << endl;
    cout << endl;

    cout << "## Result evaluation" << endl;
    cout << endl;

    // WRITE RESULT IF WANTED //
    if (writeOnlyLast) {
        cout << "  - Writing last timestep to file '" << fout << "'..." << flush;
        fstream fs (fout, fstream::out | fstream::app);
        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j < N - 1; ++j) {
                fs << setprecision(FILEOUTPUT_PRECISION) << (*all_h[i])[j] << ' ';
            }
            fs << setprecision(FILEOUTPUT_PRECISION) << (*all_h[i])[N - 1] << endl;
        }
        fs.close();
        cout << "[OK]" << endl;
    }

    // CHECK IF RESULTS ARE EQUAL //
    if (checkResult) {
        cout << "  - Verifying GPU results with CPU results:" << endl;
        cout << "    - pos x error = " << relDiffEl(pos_x_h, pos_x_gpures, N) * 100 << " %" << endl;
        cout << "    - pos y error = " << relDiffEl(pos_y_h, pos_y_gpures, N) * 100 << " %" << endl;
        cout << "    - pos z error = " << relDiffEl(pos_z_h, pos_z_gpures, N) * 100 << " %" << endl;
        cout << "    - vel x error = " << relDiffEl(vel_x_h, vel_x_gpures, N) * 100 << " %" << endl;
        cout << "    - vel y error = " << relDiffEl(vel_y_h, vel_y_gpures, N) * 100 << " %" << endl;
        cout << "    - vel z error = " << relDiffEl(vel_z_h, vel_z_gpures, N) * 100 << " %" << endl;
    }

    // COMPARE AGAINST EXTERNAL RESULTS //
    if (!fcheckfile.empty()) {
        cout << "  - Reading results from external file '" << fcheckfile << "'..." << flush;
        vector<float> pos_x_check(N);
        vector<float> pos_y_check(N);
        vector<float> pos_z_check(N);
        vector<float> vel_x_check(N);
        vector<float> vel_y_check(N);
        vector<float> vel_z_check(N);
        vector<float> m_check(N);
        fstream fs (fcheckfile, fstream::in);
        for (int j = 0; j < N; ++j) {
            fs >> pos_x_check[j];
        }
        for (int j = 0; j < N; ++j) {
            fs >> pos_y_check[j];
        }
        for (int j = 0; j < N; ++j) {
            fs >> pos_z_check[j];
        }
        for (int j = 0; j < N; ++j) {
            fs >> vel_x_check[j];
        }
        for (int j = 0; j < N; ++j) {
            fs >> vel_y_check[j];
        }
        for (int j = 0; j < N; ++j) {
            fs >> vel_z_check[j];
        }
        for (int j = 0; j < N; ++j) {
            fs >> m_check[j];
        }
        fs.close();
        cout << "[OK]" << endl;
        cout << "  - Verifying " << (checkResult ? "CPU" : "GPU") << " results with external results:" << endl;
        cout << "    - pos x error = " << relDiffEl(pos_x_check, pos_x_h, N) * 100 << " %" << endl;
        cout << "    - pos y error = " << relDiffEl(pos_y_check, pos_y_h, N) * 100 << " %" << endl;
        cout << "    - pos z error = " << relDiffEl(pos_z_check, pos_z_h, N) * 100 << " %" << endl;
        cout << "    - vel x error = " << relDiffEl(vel_x_check, vel_x_h, N) * 100 << " %" << endl;
        cout << "    - vel y error = " << relDiffEl(vel_y_check, vel_y_h, N) * 100 << " %" << endl;
        cout << "    - vel z error = " << relDiffEl(vel_z_check, vel_z_h, N) * 100 << " %" << endl;
        cout << "    - masses error = " << relDiffEl(m_check, m_h, N) * 100 << " %" << endl;
    }

    // IF VERBOSE PRINT GRIDS //
    if (verbose) {
        if (checkResult) {
            cout << "  - CPU result:" << endl;
            cout << "    - pos x: ";
            printArray(pos_x_h, N); cout << endl;
            cout << "    - pos y: ";
            printArray(pos_y_h, N); cout << endl;
            cout << "    - pos z: ";
            printArray(pos_z_h, N); cout << endl;
            cout << "    - vel x: ";
            printArray(vel_x_h, N); cout << endl;
            cout << "    - vel y: ";
            printArray(vel_y_h, N); cout << endl;
            cout << "    - vel z: ";
            printArray(vel_z_h, N); cout << endl;
            cout << "  - GPU result:" << endl;
            cout << "    - pos x: ";
            printArray(pos_x_gpures, N); cout << endl;
            cout << "    - pos y: ";
            printArray(pos_y_gpures, N); cout << endl;
            cout << "    - pos z: ";
            printArray(pos_z_gpures, N); cout << endl;
            cout << "    - vel x: ";
            printArray(vel_x_gpures, N); cout << endl;
            cout << "    - vel y: ";
            printArray(vel_y_gpures, N); cout << endl;
            cout << "    - vel z: ";
            printArray(vel_z_gpures, N); cout << endl;
            cout << "    - masses: ";
            printArray(m_h, N); cout << endl;
        }
        else {
            cout << "  - GPU result:" << endl;
            cout << "    - pos x: ";
            printArray(pos_x_h, N); cout << endl;
            cout << "    - pos y: ";
            printArray(pos_y_h, N); cout << endl;
            cout << "    - pos z: ";
            printArray(pos_z_h, N); cout << endl;
            cout << "    - vel x: ";
            printArray(vel_x_h, N); cout << endl;
            cout << "    - vel y: ";
            printArray(vel_y_h, N); cout << endl;
            cout << "    - vel z: ";
            printArray(vel_z_h, N); cout << endl;
            cout << "    - masses: ";
            printArray(m_h, N); cout << endl;
        }
    }

    // FREE MEMORY //
    cout << "  - Going to free host memory..." << flush;
    try {
        for (int i = 0; i < 7; ++i) {
            delete[] *all_h[i];
        }
    } catch (...) {
        cout << "[FAILED]" << endl;
        return EXIT_FAILURE;
    }
    cout << "[OK]" << endl;
    if (checkResult) {
        cout << "  - Going to free host memory needed for GPU result check..." << flush;
        try {
            for (int i = 0; i < 7; ++i) {
                delete[] *all_gpures[i];
            }
        } catch (...) {
            cout << "[FAILED]" << endl;
            return EXIT_FAILURE;
        }
        cout << "[OK]" << endl;
    }

    // FREE DEVICE MEMORY //
    cout << "  - Going to free device memory..." << flush;
    for (int i = 0; i < 7; ++i) {
        if (cudaFree(*all_d[i]) != cudaSuccess) {
            cout << "[FAILED]" << endl;
            exit(1);
        }
    }
    cout << "[OK]" << endl;
    cout << endl;

    // REPORT //
    cout << "## Program Report" << endl;
    cout << endl;
    cout << "  - host to device copy time = " << t_htod.count() << " s" << endl
         << "  - device to host copy time = " << t_dtoh.count() << " s" << endl
         << "  - kernel time = " << t_kernel.count() << " s" << endl;

    return EXIT_SUCCESS;
}
