#include "Timer.h"

Timer::Timer(const std::string& object_name):
    d_object_name(object_name)
{
    d_start = 0.0;
    d_stop = 0.0;
    d_time = 0.0;
    d_total = 0.0;
    d_count = 0;
}

Timer::~Timer()
{}

void Timer::Start()
{
    d_time = 0.0;
    PetscTime(&d_start);
}

void Timer::Stop()
{
    PetscTime(&d_stop);
    d_time = d_stop-d_start;
    d_total += d_time;
    d_count++;
}

void Timer::Clear()
{
    d_time = 0;
    d_total = 0;
    d_count = 0;
}

void Timer::Print(FILE *fp)
{
    PetscFPrintf(PETSC_COMM_WORLD, fp, "%s:   %f",
                 d_object_name.c_str(), d_time);
}

void Timer::PrintTotal(FILE *fp)
{
    PetscFPrintf(PETSC_COMM_WORLD, fp, "%s:   %f    %f    %d",
                 d_object_name.c_str(), d_total, getAverageTime(), d_count);
}

