
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_IOLETS_BOUNDARYCOMMS_H
#define HEMELB_LB_IOLETS_BOUNDARYCOMMS_H

#include "geometry/LatticeData.h"
#include "lb/SimulationState.h"
#include "util/UnitConverter.h"
#include "lb/iolets/BoundaryCommunicator.h"

namespace hemelb
{
  namespace lb
  {
    namespace iolets
    {

      class BoundaryComms
      {
        public:
          BoundaryComms(SimulationState* iSimState, int centreRank, const BoundaryCommunicator& boundaryComm);
          ~BoundaryComms();

          void Wait();

          // It is up to the caller to make sure only BCproc calls send
          void Send(distribn_t* density);
          void Receive(distribn_t* density);

          int GetNumProcs() const
          {
            return nProcs;
          }
          const BoundaryCommunicator& GetCommunicator() const
          {
            return bcComm;
          }

          void ReceiveDoubles(double* double_array, int size);
          void WaitAllComms();
          void FinishSend();

        private:
          // This is necessary to support BC proc having fluid sites
          bool hasBoundary;

          int nProcs;

          BoundaryCommunicator bcComm;

          MPI_Request *sendRequest;
          MPI_Status *sendStatus;

          MPI_Request receiveRequest;
          MPI_Status receiveStatus;

          SimulationState* mState;
      };

    }
  }
}

#endif /* HEMELB_LB_IOLETS_BOUNDARYCOMMS_H */
