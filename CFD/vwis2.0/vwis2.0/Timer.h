#ifndef included_Timer
#define included_Timer

#include <string>
#include <stdio.h>
#include "petsctime.h"

class Timer
{
public:
    Timer(const std::string& object_name);
    ~Timer();
    void Start();
    void Stop();
    void Clear();
    void Print(FILE *fp);
    void PrintTotal(FILE *fp);
    PetscReal getTime() {return d_time;}
    PetscReal getTotalTime() {return d_total;}
    PetscReal getAverageTime() {return d_total/ (PetscReal) d_count;}
    std::string getName() {return d_object_name;}

private:
    std::string d_object_name;
    PetscReal d_start;
    PetscReal d_stop;
    PetscReal d_time;
    PetscReal d_total;
    PetscInt d_count;
};

#endif
