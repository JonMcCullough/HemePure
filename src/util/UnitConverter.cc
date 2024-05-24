
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include "util/UnitConverter.h"
#include "constants.h"

namespace hemelb
{
  namespace util
  {

    UnitConverter::UnitConverter(PhysicalTime timeStep, PhysicalDistance voxelSize,
                                 PhysicalPosition latticeOrigin, double density) :
      latticeDistance(voxelSize), latticeTime(timeStep),
          latticeMass(density * voxelSize * voxelSize * voxelSize),
          latticeSpeed(voxelSize / latticeTime),
          latticeOrigin(latticeOrigin),
          latticePressure(latticeMass / (latticeDistance * latticeTime * latticeTime))
    {
    }

    LatticePressure UnitConverter::ConvertPressureToLatticeUnits(PhysicalPressure pressure) const
    {
      return Cs2 + ConvertPressureDifferenceToLatticeUnits(pressure - REFERENCE_PRESSURE_mmHg);
    }

    PhysicalPressure UnitConverter::ConvertPressureToPhysicalUnits(LatticePressure pressure) const
    {
      return REFERENCE_PRESSURE_mmHg + ConvertPressureDifferenceToPhysicalUnits(pressure - Cs2);
    }

    LatticePressure UnitConverter::ConvertPressureDifferenceToLatticeUnits(
                                                                           PhysicalPressure pressure_diff) const
    {
      return pressure_diff * mmHg_TO_PASCAL / latticePressure;
    }

    PhysicalPressure UnitConverter::ConvertPressureDifferenceToPhysicalUnits(
                                                                             LatticePressure pressure_diff) const
    {
      return pressure_diff * latticePressure / mmHg_TO_PASCAL;
    }

    PhysicalReciprocalTime UnitConverter::ConvertShearRateToPhysicalUnits(
                                                                          LatticeReciprocalTime shearRate) const
    {
      return shearRate / latticeTime;
    }
    LatticeDistance UnitConverter::ConvertDistanceToLatticeUnits(const PhysicalDistance& x) const
    {
      return x / latticeDistance;
    }
    PhysicalDistance UnitConverter::ConvertDistanceToPhysicalUnits(const LatticeDistance& x) const
    {
      return x * latticeDistance;
    }
    LatticePosition UnitConverter::ConvertPositionToLatticeUnits(const PhysicalPosition& x) const
    {
      return (x - latticeOrigin) / latticeDistance;
    }
    PhysicalPosition UnitConverter::ConvertPositionToPhysicalUnits(const LatticePosition& x) const
    {
      return x * latticeDistance + latticeOrigin;
    }

    LatticeSpeed UnitConverter::ConvertSpeedToLatticeUnits(const PhysicalSpeed& v) const
    {
      return v / latticeSpeed;
    }
    PhysicalSpeed UnitConverter::ConvertSpeedToPhysicalUnits(const LatticeSpeed& v) const
    {
      return v * latticeSpeed;
    }
    LatticeTime UnitConverter::ConvertTimeToLatticeUnits(const PhysicalTime& t) const
    {
      return t / latticeTime;
    }
    PhysicalTime UnitConverter::ConvertTimeToPhysicalUnits(const LatticeTime& t) const
    {
      return t * latticeTime;
    }
    PhysicalTime UnitConverter::ConvertTimeStepToPhysicalUnits(LatticeTimeStep time_step) const
    {
      return LatticeTime(time_step) * latticeTime;
    }

    distribn_t UnitConverter::ConvertAreaToLatticeUnits(const distribn_t& r) const
    {
      return r / (latticeDistance*latticeDistance);
    }
    
    distribn_t UnitConverter::ConvertAreaToPhysicalUnits(const distribn_t& r) const
    {
      return r * (latticeDistance*latticeDistance);
    }
    
    distribn_t UnitConverter::ConvertResistanceToLatticeUnits(const distribn_t& R) const
    {
      return R * ((latticeTime*latticeDistance)/BLOOD_DENSITY_Kg_per_m3);
    }
    
    distribn_t UnitConverter::ConvertResistanceToPhysicalUnits(const distribn_t& R) const
    {
      return R * (BLOOD_DENSITY_Kg_per_m3/(latticeTime*latticeDistance));
    }
    
    distribn_t UnitConverter::ConvertCapacitanceToLatticeUnits(const distribn_t& C) const
    {
      return C * (BLOOD_DENSITY_Kg_per_m3/(latticeTime*latticeTime*latticeDistance));
    }
    
    distribn_t UnitConverter::ConvertCapacitanceToPhysicalUnits(const distribn_t& C) const
    {
      return C * ((latticeTime*latticeTime*latticeDistance)/BLOOD_DENSITY_Kg_per_m3);
    }


  }
}
