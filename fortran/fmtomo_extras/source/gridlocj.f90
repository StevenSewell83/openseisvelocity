!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! PROGRAM GRIDLOC
! Relocates sources in a velocity
! field defined by fm3d. Uses a
! simple grid-search approach
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
PROGRAM gridloc
IMPLICIT NONE
INTEGER :: i,ii,j,k,l,idm,ni
INTEGER :: mrt,nr,nrad,nlat,nlon,ns,nsl,nslt
INTEGER :: irad,ilat,ilon,sstat
INTEGER :: nnth,nnph,ntt,absm,nlr
INTEGER :: iradl,ilatl,ilonl,iradlc,ilatlc,ilonlc
INTEGER :: nrp,nrs,psuse,aee,npicks
INTEGER, DIMENSION(:), ALLOCATABLE :: idtt,idsrc,idph,idp,nslr,idrn
INTEGER, DIMENSION(:,:), ALLOCATABLE :: nss,idtto
REAL :: otp,mxh,mxd,motp,dx,dy,dz
REAL :: drad,dlat,dlon,grad,glat,glon
REAL :: objf,objfm,otav,mtav,srad,slat,slon
REAL :: rd1,rd2,rd3,stw,spw,ch95,rdist,obg,obgo,rdist1,obgd
REAL :: edep,elat,elon,evlat
REAL, DIMENSION(:), ALLOCATABLE :: ttold,terr,ttp,ttpl
REAL, DIMENSION(:,:), ALLOCATABLE :: nsrad,nslat,nslon
REAL, DIMENSION(:), ALLOCATABLE :: osrad,oslat,oslon
REAL, DIMENSION(:), ALLOCATABLE :: rlat,rlon,rdep
REAL, DIMENSION(:,:), ALLOCATABLE :: tto,tte
REAL, DIMENSION(:,:,:), ALLOCATABLE :: objfg
REAL, DIMENSION(:,:,:,:), ALLOCATABLE :: ttm
REAL, DIMENSION(2,2,2) :: tr
REAL, PARAMETER :: rearth=6371.0,rtol=1.0e-4,pi=3.141592653589793
CHARACTER(LEN=30) :: srcf,argf,interf,otimef,recf,efile
CHARACTER(LEN=30) :: argfs,interfs,otimefs,recfs,psrcf,ssrcf
!
! srcf = file containing source-receiver associations
! argf = file containing grid of model times
! interf = file containing interface information
! mrt = minimum number of source-receiver combinations for relocation
! nr = Number of receivers
! tto = Observed traveltimes
! tte = Error/uncertainty associated with tto
! ttm = Model traveltimes on a grid
! nrad,nlat,nlon = Number of nodes in radius, latitude and longitude
! drad,dlat,dlon = Node spacing in radius, latitude and longitude
! grad,glat,glon = Grid origin in radius, latitude and longitude
! ns = number of sources
! nsl = number of sources per receiver
! objf = objective function
! irad,ilat,ilon = Location of objective function minimum
! objfm = minimum of objective function
! otav = observed time average
! mtav = model time average
! sstat = status of source (0=relocate; 1=no relocate)
! rearth = Radius of the earth
! srad,slat,slon = grid locaton of source
! otp = origin time perturbation
! mxh = maximum height of model region
! mxd = maximum depth of model region
! ni = number of interfaces
! nnth,nnph = number of interfaces nodes in lat and long
! motp = maximum origin time perturbation allowed
! otimef = observed time file
! recf = receiver file
! nsrad,nslat,nslon = new source locations
! nss = new source status
! osrad,oslat,oslon = Old source location
! ttold = Old traveltimes
! terr = traveltime uncertainty
! idtt,idsrc,idph,idp = Integer variables for output
! ntt = Number of traveltimes
! idtto = source id of tto
! nslr = number of sources per receiver
! absm = apply bisection refinement to source location
! nlr = number of levels of refinement
! ttp = model traveltime at a given point
! tr = traveltime values for trilinear interpolation
! dx,dy,dz = displacements for trilienear interpolation
! iradl,ilatl,ilonl = indices relative to irad,ilat,ilon
! iradlc,ilatlc,ilonlc = indices relative to iradl,ilatl,ilonl
! ttpl = ttp for optimum point
! argfs,interfs,otimefs,recfs = as for argf,interf,otimef,recf bbut for S-waves
! nrp,nrs = number of receivers for P and S
! stw = Weighting used to trade-off between P and S importance
! psuse = Use P-times (1), P and S (2) or P, S and S-P (3)  for relocation
! psrcf = Source file for P
! ssrcf = Source file for S
! idrn = Id of receiver number
! rlat,rlon,rdep = Location of receivers
! spw = Weight for S-P times
! efile = output error file
! aee = apply error ellipse (0=no, 1=yes)
! ch95 = Chi-squared statistic for a p-value of 0.05 and three degrees of freedom.
! npicks = Number of picks that are used in objective function
! objfg = Value of objective function on entire grid
! rdist,rdist1 = distance in node separation units from source to 95% ci.
! obg,obgo,obgd = objective function values
! evlat = event latitude
!
! Set parameters
!
ch95=7.815
OPEN(UNIT=10,FILE='gridlocj.in',STATUS='old')
READ(10,1)srcf
READ(10,1)argf
READ(10,1)argfs
READ(10,1)psrcf
READ(10,1)ssrcf
READ(10,*)mrt
READ(10,*)motp
READ(10,1)interf
READ(10,1)otimef
READ(10,1)otimefs
READ(10,1)recf
READ(10,1)recfs
READ(10,*)absm
READ(10,*)nlr
READ(10,*)aee
READ(10,1)efile
READ(10,*)stw
READ(10,*)spw
READ(10,*)psuse
1 FORMAT(a30)
CLOSE(10)
IF(nlr.LE.1)THEN
   WRITE(6,*)'Value assigned to sub-cell dicing must be'
   WRITE(6,*)'greater than 1'
   WRITE(6,*)'Terminating program!!!'
   STOP
ENDIF
!
! First, read in srcf, and determine how many
! sources there are to relocate
!
OPEN(UNIT=20,FILE=srcf,STATUS='old')
READ(20,*)nr
ns=0
DO i=1,nr
   READ(20,*)idm,nsl
   DO j=1,nsl
      READ(20,*)idm
      IF(ns.LT.idm)ns=idm
   ENDDO
ENDDO
CLOSE(20)
ALLOCATE(tto(nr,ns))
ALLOCATE(tte(nr,ns))
ALLOCATE(idtto(nr,ns))
ALLOCATE(nss(nr,ns))
ALLOCATE(nsrad(nr,ns),nslat(nr,ns),nslon(nr,ns))
ALLOCATE(idrn(nr))
tto=-10.0
nss=1
!
! Determine how many unique receiver locations there are
!
ALLOCATE(rdep(nr),rlat(nr),rlon(nr))
OPEN(UNIT=10, FILE=psrcf,STATUS='old')
READ(10,*)nrp
DO i=1,nrp
   READ(10,*)
   READ(10,*)rdep(i),rlat(i),rlon(i),idrn(i)
   READ(10,*)ii
   DO j=1,ii
      READ(10,*)
      READ(10,*)
      READ(10,*)
   ENDDO
ENDDO
CLOSE(10)
OPEN(UNIT=10, FILE=ssrcf,STATUS='old')
READ(10,*)nrs
DO i=nrp+1,nr
   READ(10,*)
   READ(10,*)rdep(i),rlat(i),rlon(i),idrn(i)
   READ(10,*)ii
   DO j=1,ii
      READ(10,*)
      READ(10,*)
      READ(10,*)
   ENDDO
ENDDO
CLOSE(10)
!
! Work out the location of the bounding interfaces
! for P
!
OPEN(UNIT=10,FILE=interf,STATUS='old')
READ(10,*)ni
READ(10,*)nnth,nnph
READ(10,*)
READ(10,*)
DO i=1,ni
   DO j=1,nnth
      DO k=1,nnph
         READ(10,*)mxd
      ENDDO
   ENDDO
   IF(i.EQ.1)mxh=mxd
ENDDO
mxh=rearth-mxh
mxd=rearth-mxd
CLOSE(10)
!
! Now read in the model traveltime grid for P
!
OPEN(UNIT=30,FILE=argf,STATUS='old')
READ(30,*)nrad,nlat,nlon
READ(30,*)drad,dlat,dlon
READ(30,*)grad,glat,glon
READ(30,*)nrp
IF(nrp.GE.nr)THEN
   WRITE(6,*)'ERROR!! Number of receivers for P'
   WRITE(6,*)'is at least equal to the total number'
   WRITE(6,*)'of receivers!!!'
   WRITE(6,*)'TERMINATING PROGRAM'
   STOP
ENDIF
ALLOCATE(ttm(nr,nrad,nlat,nlon))
DO i=1,nrp
   READ(30,*)
   DO j=1,nlon
      DO k=1,nlat
         DO l=1,nrad
            READ(30,*)ttm(i,l,k,j)
         ENDDO
      ENDDO
   ENDDO
ENDDO
CLOSE(30)
!
! Now append the model traveltime grid for S
!
OPEN(UNIT=30,FILE=argfs,STATUS='old')
READ(30,*)rd1,rd2,rd3
IF(rd1.NE.nrad.OR.rd2.NE.nlat.OR.rd3.NE.nlon)THEN
   WRITE(6,*)'ERROR!!!!!!!'
   WRITE(6,*)'Number of propagation grid nodes in each dimension'
   WRITE(6,*)'must be identical for P and S!'
   WRITE(6,*)'TERMINATING PROGRAM'
   STOP
ENDIF
READ(30,*)rd1,rd2,rd3
IF(ABS(rd1-drad).GT.rtol.OR.ABS(rd2-dlat).GT.rtol.OR.ABS(rd3-dlon).GT.rtol)THEN
   WRITE(6,*)'ERROR!!!!!!!'
   WRITE(6,*)'Propagation grid spacing between nodes'
   WRITE(6,*)'must be identical for P and S in all dimensions!'
   WRITE(6,*)'TERMINATING PROGRAM'
   STOP
ENDIF
READ(30,*)rd1,rd2,rd3
IF(ABS(rd1-grad).GT.rtol.OR.ABS(rd2-glat).GT.rtol.OR.ABS(rd3-glon).GT.rtol)THEN
   WRITE(6,*)'ERROR!!!!!!!'
   WRITE(6,*)'Propagation grid origin'
   WRITE(6,*)'must be identical for P and S models!'
   WRITE(6,*)'TERMINATING PROGRAM'
   STOP
ENDIF
READ(30,*)nrs
IF(nrp+nrs.NE.nr)THEN
   WRITE(6,*)'ERROR!! Number of receivers for P'
   WRITE(6,*)'and S must equal total number'
   WRITE(6,*)'of receivers!!!'
   WRITE(6,*)'TERMINATING PROGRAM'
   STOP
ENDIF
DO i=nrp+1,nr
   READ(30,*)
   DO j=1,nlon
      DO k=1,nlat
         DO l=1,nrad
            READ(30,*)ttm(i,l,k,j)
         ENDDO
      ENDDO
   ENDDO
ENDDO
CLOSE(30)
!
! Now populate the observed traveltime matrix
!
ALLOCATE(nslr(nr))
OPEN(UNIT=20,FILE=srcf,STATUS='old')
READ(20,*)idm
DO i=1,nr
   READ(20,*)idm,nslr(i)
   DO j=1,nslr(i)
      READ(20,*)idtto(i,j),tto(i,idtto(i,j)),tte(i,idtto(i,j))
      IF(i.GT.nrp)tte(i,idtto(i,j))=stw*tte(i,idtto(i,j))
   ENDDO
ENDDO
CLOSE(20)
!
! Now loop through every source, compute objective function over
! the entire grid, and locate the minimum
!
ALLOCATE(objfg(nrad,nlat,nlon))
OPEN(UNIT=60,FILE=efile,STATUS='unknown')
IF(absm.EQ.1)ALLOCATE(ttp(nr),ttpl(nr))
DO i=1,ns
!
!  Now compute objective function for every grid point
!
   objfm=1.0e20
   sstat=0
   DO j=1,nrad
      DO k=1,nlat
         DO l=1,nlon
            otav=0.0
            mtav=0.0
            nsl=0
            npicks=0
!
!           Find average P (and S if required)
!
            idm=nrp
            IF(psuse.EQ.4)idm=nr
            DO ii=1,idm
               IF(tto(ii,i).GT.0.)THEN
                  otav=otav+tto(ii,i)
                  IF(ii.LE.nrp)THEN
                     nsl=nsl+1
                  ENDIF
                  mtav=mtav+ttm(ii,j,k,l)
               ENDIF
            ENDDO
!
!           Now add unique S if it exists
!
            IF(psuse.NE.1.AND.psuse.NE.4)THEN
               DO ii=nrp+1,nr
                  IF(tto(ii,i).GT.0.)THEN
                     IF(idrn(ii).GT.nrp)THEN
                        otav=otav+tto(ii,i)
                        nsl=nsl+1
                        mtav=mtav+ttm(ii,j,k,l)
                     ENDIF
                  ENDIF
               ENDDO
            ENDIF
            IF(nsl.LT.mrt)THEN
               sstat=1
               EXIT
            ENDIF
            otav=otav/nsl
            mtav=mtav/nsl
            objf=0.0
!
!           Objective function for P (and S if required)
!
            idm=nrp
            IF(psuse.EQ.4)idm=nr
            DO ii=1,idm
               IF(tto(ii,i).GT.0.)THEN
                  objf=objf+((tto(ii,i)-otav)-(ttm(ii,j,k,l)-mtav))**2/tte(ii,i)**2
                  npicks=npicks+1
               ENDIF
            ENDDO
!
!           Add objective functions for S if required
!
            IF(psuse.NE.1.AND.psuse.NE.4)THEN
               DO ii=nrp+1,nr
                  IF(tto(ii,i).GT.0.)THEN
                     IF(idrn(ii).GT.nrp)THEN
                        objf=objf+((tto(ii,i)-otav)-(ttm(ii,j,k,l)-mtav))**2/tte(ii,i)**2
                        npicks=npicks+1
                     ENDIF
                  ENDIF
               ENDDO
            ENDIF
!
!           Now add objective function for S-P if required
!
            IF(psuse.EQ.3)THEN
               DO ii=nrp+1,nr
                  IF(tto(ii,i).GT.0.)THEN
                     IF(idrn(ii).LE.nrp)THEN
                        IF(tto(idrn(ii),i).GT.0.)THEN
                           rd1=tto(ii,i)-tto(idrn(ii),i)
                           rd2=ttm(ii,j,k,l)-ttm(idrn(ii),j,k,l)
                           objf=objf+spw*(rd1-rd2)**2
                           npicks=npicks+1
                        ENDIF
                     ENDIF
                  ENDIF
               ENDDO
            ENDIF
            objfg(j,k,l)=objf
            IF(objf.LT.objfm)THEN
               objfm=objf
               irad=j
               ilat=k
               ilon=l
            ENDIF
         ENDDO
         IF(sstat.EQ.1)EXIT
      ENDDO
      IF(sstat.EQ.1)EXIT
   ENDDO
!
!  Now that we have determined the grid point closest to the source, we can 
!  compute the 95 percent confidence interval (in units of km).
   IF(sstat.EQ.0)THEN
!
!     First, compute confidence interval in depth
!
      obgd=objfg(irad,ilat,ilon)*ch95/real(npicks-3)
      rdist=0.0
      obgo=0.0
      DO j=irad+1,nrad
         obg=ABS(objfg(irad,ilat,ilon)-objfg(j,ilat,ilon))
         IF(obg.GT.obgd)THEN
            rdist=(j-1)-irad+(obgd-obgo)/(obg-obgo)
            EXIT
         ENDIF
         obgo=obg
      ENDDO
      rdist1=0.0
      obgo=0.0
      DO j=irad-1,1,-1
         obg=ABS(objfg(irad,ilat,ilon)-objfg(j,ilat,ilon))
         IF(obg.GT.obgd)THEN
            rdist1=irad-(j+1)+(obgd-obgo)/(obg-obgo)
            EXIT
         ENDIF
         obgo=obg
      ENDDO
      IF(rdist1.gt.rdist)rdist=rdist1
!
!     Convert grid spacing value to km
!
      edep=rdist*drad
!
!     Now repeat for latitude
!
      rdist=0.0
      obgo=0.0
      DO j=ilat+1,nlat
         obg=ABS(objfg(irad,ilat,ilon)-objfg(irad,j,ilon))
         IF(obg.GT.obgd)THEN
            rdist=(j-1)-ilat+(obgd-obgo)/(obg-obgo)
            EXIT
         ENDIF
         obgo=obg
      ENDDO
      rdist1=0.0
      obgo=0.0
      DO j=ilat-1,1,-1
         obg=ABS(objfg(irad,ilat,ilon)-objfg(irad,j,ilon))
         IF(obg.GT.obgd)THEN
            rdist1=ilat-(j+1)+(obgd-obgo)/(obg-obgo)
            EXIT
         ENDIF
         obgo=obg
      ENDDO
      IF(rdist1.gt.rdist)rdist=rdist1
!
!     Convert grid spacing value to km
!
      elat=rdist*dlat*2.0*pi*rearth/360.0
!
!     Now repeat for longitude
!
      rdist=0.0
      obgo=0.0
      DO j=ilon+1,nlon
         obg=ABS(objfg(irad,ilat,ilon)-objfg(irad,ilat,j))
         IF(obg.GT.obgd)THEN
            rdist=(j-1)-ilon+(obgd-obgo)/(obg-obgo)
            EXIT
         ENDIF
         obgo=obg
      ENDDO
      rdist1=0.0
      obgo=0.0
      DO j=ilon-1,1,-1
         obg=ABS(objfg(irad,ilat,ilon)-objfg(irad,ilat,j))
         IF(obg.GT.obgd)THEN
            rdist1=ilon-(j+1)+(obgd-obgo)/(obg-obgo)
            EXIT
         ENDIF
         obgo=obg
      ENDDO
      IF(rdist1.gt.rdist)rdist=rdist1
!
!     Convert grid spacing value to km
!
      evlat=glat+(ilat-1)*dlat
      elon=rdist*dlat*2.0*pi*rearth*cos(evlat*pi/180.)/360.0
   ENDIF
   IF(sstat.eq.1)THEN
      edep=0.0
      elat=0.0
      elon=0.0
   ENDIF
!
!  Determine location in depth, latitude and longitude
!
   IF(sstat.EQ.0)THEN
!
!     If necessary, attempt to refine source location
!     using bilinear interpolation
!
      IF(absm.EQ.1)THEN
!
!        To start, compute the objective function at the centre
!        of each of the eight surrounding cells, and see which is
!        a minimum.
!
         objfm=1.0e20
         DO j=-1,1,2
            IF(irad+j.LT.1.OR.irad+j.GT.nrad)CYCLE
            DO k=-1,1,2
               IF(ilat+k.LT.1.OR.ilat+k.GT.nlat)CYCLE
               DO l=-1,1,2
                  IF(ilon+l.LT.1.OR.ilon+l.GT.nlon)CYCLE
                  otav=0.0
                  mtav=0.0
                  nsl=0
                  idm=nrp
                  IF(psuse.EQ.4)idm=nr
                  DO ii=1,idm
                     IF(tto(ii,i).GT.0.)THEN
                        otav=otav+tto(ii,i)
                        nsl=nsl+1
!
!                       Compute model traveltime in cell centre using trilinear interpolation
! 
                        tr(1,1,1)=ttm(ii,irad,ilat,ilon)
                        tr(2,1,1)=ttm(ii,irad+j,ilat,ilon) 
                        tr(1,2,1)=ttm(ii,irad,ilat+k,ilon)
                        tr(2,2,1)=ttm(ii,irad+j,ilat+k,ilon)
                        tr(1,1,2)=ttm(ii,irad,ilat,ilon+l)
                        tr(2,1,2)=ttm(ii,irad+j,ilat,ilon+l)
                        tr(1,2,2)=ttm(ii,irad,ilat+k,ilon+l) 
                        tr(2,2,2)=ttm(ii,irad+j,ilat+k,ilon+l)
                        dx=0.5*drad
                        dy=0.5*dlat
                        dz=0.5*dlon
                        CALL trilinear(tr,dx,dy,dz,drad,dlat,dlon,rd1)
                        ttp(ii)=rd1
                        mtav=mtav+ttp(ii)
                     ENDIF
                  ENDDO
!
!                 Now add unique S if it exists
!
                  IF(psuse.NE.1.AND.psuse.NE.4)THEN
                     DO ii=nrp+1,nr
                        IF(tto(ii,i).GT.0.)THEN
!
!                          Compute model traveltime in cell centre
!
                           tr(1,1,1)=ttm(ii,irad,ilat,ilon)
                           tr(2,1,1)=ttm(ii,irad+j,ilat,ilon)
                           tr(1,2,1)=ttm(ii,irad,ilat+k,ilon)
                           tr(2,2,1)=ttm(ii,irad+j,ilat+k,ilon)
                           tr(1,1,2)=ttm(ii,irad,ilat,ilon+l)
                           tr(2,1,2)=ttm(ii,irad+j,ilat,ilon+l)
                           tr(1,2,2)=ttm(ii,irad,ilat+k,ilon+l)
                           tr(2,2,2)=ttm(ii,irad+j,ilat+k,ilon+l)
                           dx=0.5*drad
                           dy=0.5*dlat
                           dz=0.5*dlon
                           CALL trilinear(tr,dx,dy,dz,drad,dlat,dlon,rd1)
                           ttp(ii)=rd1
                           IF(idrn(ii).GT.nrp)THEN
                              otav=otav+tto(ii,i)
                              nsl=nsl+1
                              mtav=mtav+ttp(ii)
                           ENDIF
                         ENDIF
                     ENDDO
                  ENDIF
                  otav=otav/nsl
                  mtav=mtav/nsl
                  objf=0.0
                  idm=nrp
                  IF(psuse.EQ.4)idm=nr
                  DO ii=1,idm
                     IF(tto(ii,i).GT.0.)THEN
                        objf=objf+((tto(ii,i)-otav)-(ttp(ii)-mtav))**2/tte(ii,i)**2
                     ENDIF
                  ENDDO
!
!                 Add objective functions for S if required
!
                  IF(psuse.NE.1.AND.psuse.NE.4)THEN
                     DO ii=nrp+1,nr
                        IF(tto(ii,i).GT.0.)THEN
                           IF(idrn(ii).GT.nrp)THEN
                              objf=objf+((tto(ii,i)-otav)-(ttp(ii)-mtav))**2/tte(ii,i)**2
                           ENDIF
                        ENDIF
                     ENDDO
                  ENDIF
!
!                 Now add objective function for S-P if required
!
                  IF(psuse.EQ.3)THEN
                     DO ii=nrp+1,nr
                        IF(tto(ii,i).GT.0.)THEN
                           IF(idrn(ii).LE.nrp)THEN
                              IF(tto(idrn(ii),i).GT.0.)THEN
                                 rd1=tto(ii,i)-tto(idrn(ii),i)
                                 rd2=ttp(ii)-ttp(idrn(ii))
                                 objf=objf+spw*(rd1-rd2)**2
                              ENDIF
                           ENDIF
                        ENDIF
                     ENDDO
                  ENDIF
                  IF(objf.LT.objfm)THEN
                     objfm=objf
                     iradl=j
                     ilatl=k
                     ilonl=l
                  ENDIF
               ENDDO
            ENDDO
         ENDDO
!
!        Now we have located the cell in which the minimum resides. Locate minimum using a basic
!        grid search
!
         objfm=1.0e20
         DO j=1,nlr
            DO k=1,nlr
               DO l=1,nlr
                  otav=0.0
                  mtav=0.0
                  nsl=0
                  idm=nrp
                  IF(psuse.EQ.4)idm=nr
                  DO ii=1,idm
                     IF(tto(ii,i).GT.0.)THEN
                        otav=otav+tto(ii,i)
                        nsl=nsl+1
!
!                       Compute model traveltime in cell using trilinear interpolation
!
                        tr(1,1,1)=ttm(ii,irad,ilat,ilon)
                        tr(2,1,1)=ttm(ii,irad+iradl,ilat,ilon)
                        tr(1,2,1)=ttm(ii,irad,ilat+ilatl,ilon)
                        tr(2,2,1)=ttm(ii,irad+iradl,ilat+ilatl,ilon)
                        tr(1,1,2)=ttm(ii,irad,ilat,ilon+ilonl)
                        tr(2,1,2)=ttm(ii,irad+iradl,ilat,ilon+ilonl)
                        tr(1,2,2)=ttm(ii,irad,ilat+ilatl,ilon+ilonl)
                        tr(2,2,2)=ttm(ii,irad+iradl,ilat+ilatl,ilon+ilonl)
                        dx=(REAL(j-1)/(nlr-1))*drad
                        dy=(REAL(k-1)/(nlr-1))*dlat
                        dz=(REAL(l-1)/(nlr-1))*dlon
                        CALL trilinear(tr,dx,dy,dz,drad,dlat,dlon,rd1)
                        ttp(ii)=rd1
                        mtav=mtav+ttp(ii)
                     ENDIF
                  ENDDO
!
!                 Now add unique S if it exists
!
                  IF(psuse.NE.4)THEN
                     DO ii=nrp+1,nr
                        IF(tto(ii,i).GT.0.)THEN
!
!                          Compute model traveltime in cell centre
!
                           tr(1,1,1)=ttm(ii,irad,ilat,ilon)
                           tr(2,1,1)=ttm(ii,irad+iradl,ilat,ilon)
                           tr(1,2,1)=ttm(ii,irad,ilat+ilatl,ilon)
                           tr(2,2,1)=ttm(ii,irad+iradl,ilat+ilatl,ilon)
                           tr(1,1,2)=ttm(ii,irad,ilat,ilon+ilonl)
                           tr(2,1,2)=ttm(ii,irad+iradl,ilat,ilon+ilonl)
                           tr(1,2,2)=ttm(ii,irad,ilat+ilatl,ilon+ilonl)
                           tr(2,2,2)=ttm(ii,irad+iradl,ilat+ilatl,ilon+ilonl)
                           dx=(REAL(j-1)/(nlr-1))*drad
                           dy=(REAL(k-1)/(nlr-1))*dlat
                           dz=(REAL(l-1)/(nlr-1))*dlon
                           CALL trilinear(tr,dx,dy,dz,drad,dlat,dlon,rd1)
                           ttp(ii)=rd1
                           IF(idrn(ii).GT.nrp.AND.psuse.NE.1)THEN
                              otav=otav+tto(ii,i)
                              nsl=nsl+1
                              mtav=mtav+ttp(ii)
                           ENDIF
                        ENDIF
                     ENDDO
                  ENDIF
                  otav=otav/nsl
                  mtav=mtav/nsl
                  objf=0.0
                  idm=nrp
                  IF(psuse.EQ.4)idm=nr
                  DO ii=1,idm
                     IF(tto(ii,i).GT.0.)THEN
                        objf=objf+((tto(ii,i)-otav)-(ttp(ii)-mtav))**2/tte(ii,i)**2
                     ENDIF
                  ENDDO
!
!                 Add objective functions for S if required
!
                  IF(psuse.NE.1.AND.psuse.NE.4)THEN
                     DO ii=nrp+1,nr
                        IF(tto(ii,i).GT.0.)THEN
                           IF(idrn(ii).GT.nrp)THEN
                              objf=objf+((tto(ii,i)-otav)-(ttp(ii)-mtav))**2/tte(ii,i)**2
                           ENDIF
                        ENDIF
                     ENDDO
                  ENDIF
!
!                 Now add objective function for S-P if required
!
                  IF(psuse.EQ.3)THEN
                     DO ii=nrp+1,nr
                        IF(tto(ii,i).GT.0.)THEN
                           IF(idrn(ii).LE.nrp)THEN
                              IF(tto(idrn(ii),i).GT.0.)THEN
                                 rd1=tto(ii,i)-tto(idrn(ii),i)
                                 rd2=ttp(ii)-ttp(idrn(ii))
                                 objf=objf+spw*(rd1-rd2)**2
                              ENDIF
                           ENDIF
                        ENDIF
                     ENDDO
                  ENDIF
                  IF(objf.LT.objfm)THEN
                     objfm=objf
                     iradlc=j
                     ilatlc=k
                     ilonlc=l
                     ttpl=ttp
                  ENDIF
               ENDDO
            ENDDO
         ENDDO
!
!        Now that we have located the minimum, compute spherical coordinates
!
         srad=grad+(irad-1)*drad
         srad=rearth-srad
         slat=glat+(ilat-1)*dlat
         slon=glon+(ilon-1)*dlon
         srad=srad-iradl*(REAL(iradlc-1)/(nlr-1))*drad
         slat=slat+ilatl*(REAL(ilatlc-1)/(nlr-1))*dlat
         slon=slon+ilonl*(REAL(ilonlc-1)/(nlr-1))*dlon
!
!        Now check that bounds are appropriate
!
         IF(srad.LT.mxh)srad=mxh+1.0e-5
         IF(srad.GT.mxd)srad=mxd-1.0e-5
         IF(slat.LE.glat)slat=glat+1.0e-5
         IF(slat.GE.glat+(nlat-1)*dlat)slat=glat+(nlat-1)*dlat-1.0e-5
         IF(slon.LE.glon)slon=glon+1.0e-5
         IF(slon.GE.glon+(nlon-1)*dlon)slon=glon+(nlon-1)*dlon-1.0e-5
!
!        Finally, compute origin time perturbation 
!
         nsl=0
         otp=0.0
         idm=nrp
         IF(psuse.EQ.4)idm=nr
         DO j=1,idm
            IF(tto(j,i).GT.0.)THEN
              nsl=nsl+1
              otp=otp+(tto(j,i)-ttpl(j))
            ENDIF
         ENDDO
         otp=otp/nsl
         IF(ABS(otp).GT.motp)sstat=1
         idm=nrp
         IF(psuse.EQ.4)idm=nr
         DO j=1,idm
            IF(tto(j,i).GT.0.)THEN
               tto(j,i)=tto(j,i)-otp
               IF(tto(j,i).LE.0.001)THEN
                  tto(j,i)=1.0
                  sstat=1
               ENDIF
            ENDIF
         ENDDO
         IF(psuse.NE.4)THEN
            nsl=0
            otp=0.0
            DO j=nrp+1,nr
               IF(tto(j,i).GT.0.)THEN
                  nsl=nsl+1
                  otp=otp+(tto(j,i)-ttpl(j))
               ENDIF
            ENDDO
            otp=otp/nsl
            DO j=nrp+1,nr
               IF(tto(j,i).GT.0.)THEN
                  tto(j,i)=tto(j,i)-otp
                  IF(tto(j,i).LE.0.001)THEN
                     tto(j,i)=1.0
                     sstat=1
                  ENDIF
               ENDIF
            ENDDO
         ENDIF
      ELSE 
!
!        Check bounds
!
         srad=grad+(irad-1)*drad
         srad=rearth-srad
         slat=glat+(ilat-1)*dlat
         slon=glon+(ilon-1)*dlon
         IF(ilat.EQ.1)slat=slat+1.0e-5
         IF(ilat.EQ.nlat)slat=slat-1.0e-5
         IF(ilon.EQ.1)slon=slon+1.0e-5
         IF(ilon.EQ.nlon)slon=slon-1.0e-5
!
!        Ensure new locations lie within model bounds
!
         IF(srad.LT.mxh)THEN
            irad=irad-INT((mxh-srad)/drad)-1
         ELSE IF(srad.GT.mxd)THEN
            irad=irad+INT((srad-mxd)/drad)+1
         ENDIF
         srad=grad+(irad-1)*drad
         srad=rearth-srad
!
!        Compute origin time perturbation
!
         nsl=0
         otp=0.0
         idm=nrp
         IF(psuse.EQ.4)idm=nr
         DO j=1,idm
            IF(tto(j,i).GT.0.)THEN
              nsl=nsl+1
              otp=otp+(tto(j,i)-ttm(j,irad,ilat,ilon))
            ENDIF
         ENDDO
         otp=otp/nsl
         IF(ABS(otp).GT.motp)sstat=1
         idm=nrp
         IF(psuse.EQ.4)idm=nr
         DO j=1,idm
            IF(tto(j,i).GT.0.)THEN
               tto(j,i)=tto(j,i)-otp
               IF(tto(j,i).LE.0.001)THEN
                  tto(j,i)=1.0
                  sstat=1
               ENDIF
            ENDIF
         ENDDO
         IF(psuse.NE.4)THEN
            nsl=0
            otp=0.0
            DO j=nrp+1,nr
               IF(tto(j,i).GT.0.)THEN
                  nsl=nsl+1
                  otp=otp+(tto(j,i)-ttm(j,irad,ilat,ilon))
               ENDIF
            ENDDO
            otp=otp/nsl
            DO j=nrp+1,nr
               IF(tto(j,i).GT.0.)THEN
                  tto(j,i)=tto(j,i)-otp
                  IF(tto(j,i).LE.0.001)THEN
                     tto(j,i)=1.0
                     sstat=1
                  ENDIF
               ENDIF
            ENDDO
         ENDIF
      ENDIF
   ENDIF
!
!  Record new locations
!
   IF(sstat.EQ.0)THEN
      DO j=1,nr
         nsrad(j,i)=srad
         nslat(j,i)=slat
         nslon(j,i)=slon
         nss(j,i)=sstat
      ENDDO
   ENDIF
   IF(sstat.EQ.1)THEN
      srad=0.0
      slat=0.0
      slon=0.0
   ENDIF
   WRITE(60,*)-srad,slat,slon,edep,elat,elon
ENDDO
CLOSE(60)
!
! Write output to file. Begin by reading in the observed time
! file that needs to be modified
!
OPEN(UNIT=40,FILE=otimef,STATUS='old')
READ(40,*)ntt
ALLOCATE(idtt(ntt),idsrc(ntt),idph(ntt),idp(ntt))
ALLOCATE(ttold(ntt),terr(ntt))
DO i=1,ntt
   READ(40,*)idtt(i),idsrc(i),idph(i),idp(i),ttold(i),terr(i)
ENDDO
CLOSE(40)
OPEN(UNIT=40,FILE=otimef,STATUS='unknown')
WRITE(40,*)ntt
nsl=0
DO i=1,nrp
   DO j=1,nslr(i)
      nsl=nsl+1
      IF(nss(i,idtto(i,j)).EQ.0)ttold(nsl)=tto(i,idtto(i,j))
      WRITE(40,*)idtt(nsl),idsrc(nsl),idph(nsl),idp(nsl),ttold(nsl),terr(nsl)
   ENDDO
ENDDO
CLOSE(40)
!
! Now modify the receiver file
!
OPEN(UNIT=50,FILE=recf,STATUS='old')
READ(50,*)ntt
ALLOCATE(osrad(ntt),oslat(ntt),oslon(ntt))
DO i=1,ntt
   READ(50,*)osrad(i),oslat(i),oslon(i)
   READ(50,*)idtt(i)
   READ(50,*)idsrc(i)
   READ(50,*)idph(i)
ENDDO
CLOSE(50)
OPEN(UNIT=50,FILE=recf,STATUS='unknown')
WRITE(50,*)ntt
nsl=0
k=0
DO i=1,nrp
   DO j=1,nslr(i)
      nsl=nsl+1
      IF(nss(i,idtto(i,j)).EQ.0)THEN
         osrad(nsl)=nsrad(i,idtto(i,j))
         oslat(nsl)=nslat(i,idtto(i,j))
         oslon(nsl)=nslon(i,idtto(i,j))
         k=k+1
      ENDIF
      WRITE(50,*)osrad(nsl),oslat(nsl),oslon(nsl)
      WRITE(50,*)idtt(nsl)
      WRITE(50,*)idsrc(nsl)
      WRITE(50,*)idph(nsl)
   ENDDO
ENDDO
CLOSE(50)
nslt=nsl
DEALLOCATE(idtt,idsrc,idph,idp,ttold,terr)
DEALLOCATE(osrad,oslat,oslon)
!
! Now repeat for S
!
OPEN(UNIT=40,FILE=otimefs,STATUS='old')
READ(40,*)ntt
ALLOCATE(idtt(ntt),idsrc(ntt),idph(ntt),idp(ntt))
ALLOCATE(ttold(ntt),terr(ntt))
DO i=1,ntt
   READ(40,*)idtt(i),idsrc(i),idph(i),idp(i),ttold(i),terr(i)
ENDDO
CLOSE(40)
OPEN(UNIT=40,FILE=otimefs,STATUS='unknown')
WRITE(40,*)ntt
nsl=0
DO i=nrp+1,nr
   DO j=1,nslr(i)
      nsl=nsl+1
      IF(nss(i,idtto(i,j)).EQ.0)ttold(nsl)=tto(i,idtto(i,j))
      WRITE(40,*)idtt(nsl),idsrc(nsl),idph(nsl),idp(nsl),ttold(nsl),terr(nsl)
   ENDDO
ENDDO
CLOSE(40)
!
! Now modify receiver file for S
!
OPEN(UNIT=50,FILE=recfs,STATUS='old')
READ(50,*)ntt
ALLOCATE(osrad(ntt),oslat(ntt),oslon(ntt))
DO i=1,ntt
   READ(50,*)osrad(i),oslat(i),oslon(i)
   READ(50,*)idtt(i)
   READ(50,*)idsrc(i)
   READ(50,*)idph(i)
ENDDO
CLOSE(50)
OPEN(UNIT=50,FILE=recfs,STATUS='unknown')
WRITE(50,*)ntt
nsl=0
DO i=nrp+1,nr
   DO j=1,nslr(i)
      nsl=nsl+1
      IF(nss(i,idtto(i,j)).EQ.0)THEN
         osrad(nsl)=nsrad(i,idtto(i,j))
         oslat(nsl)=nslat(i,idtto(i,j))
         oslon(nsl)=nslon(i,idtto(i,j))
         k=k+1
      ENDIF
      WRITE(50,*)osrad(nsl),oslat(nsl),oslon(nsl)
      WRITE(50,*)idtt(nsl)
      WRITE(50,*)idsrc(nsl)
      WRITE(50,*)idph(nsl)
   ENDDO
ENDDO
nslt=nslt+nsl
CLOSE(50)
DEALLOCATE(ttm,tto,tte,nss,nsrad,nslat,nslon)
DEALLOCATE(idtt,idsrc,idph,idp)
DEALLOCATE(ttold,terr)
DEALLOCATE(osrad,oslat,oslon)
DEALLOCATE(idtto,nslr)
DEALLOCATE(idrn,objfg)
DEALLOCATE(rdep,rlat,rlon)
IF(absm.EQ.1)DEALLOCATE(ttp,ttpl)
WRITE(6,*)k,' out of ',nslt,' events relocated'
WRITE(6,*)'Program gridloc successfully completed'
END PROGRAM gridloc

SUBROUTINE TRILINEAR(tr,dx,dy,dz,drad,dlat,dlon,tt)
IMPLICIT NONE
INTEGER :: i,j,k
REAL :: dx,dy,dz,drad,dlat,dlon,tt
REAL, DIMENSION(2,2,2) :: tr
REAL, DIMENSION(2) :: dxr,dyr,dzr
tt=0.
dxr(1)=dx
dxr(2)=drad-dx
dyr(1)=dy
dyr(2)=dlat-dy
dzr(1)=dz
dzr(2)=dlon-dz
DO i=1,2
   DO j=1,2
      DO k=1,2
         tt=tt+tr(i,j,k)*(1.0-ABS(dxr(i))/drad)*(1.0-ABS(dyr(j))/dlat)*(1.0-ABS(dzr(k))/dlon)
      ENDDO
   ENDDO
ENDDO
END SUBROUTINE TRILINEAR
