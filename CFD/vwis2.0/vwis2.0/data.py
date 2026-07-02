import numpy as np
import h5py 
import os
import struct
import argparse

def main():
    parser = argparse.ArgumentParser(description='Create XDMF Files')
    parser.add_argument('--create_grid', action='store_true',
                         help='Convert grid to h5')
    parser.add_argument('--grid_xyz', action='store_true',
                         help='Grid is xyz format')
    parser.add_argument('--grid_bin', action='store_true',
                         help='Grid is binary format')
    parser.add_argument('--iterations', nargs=3, type=int,
                         help="Start, end, skip iterations")
    parser.add_argument('--average', type=int,
                         help="Average",default=0)
    parser.add_argument('--directory', help="Directory of Field results",
                        default=os.getcwd())
    parser.add_argument('--average_directory', 
                        help="Directory of average rsults",
                        default=os.getcwd())
    parser.add_argument('--time', action='store_true',
                         help='Create 1 xdmf file over time')

    args = parser.parse_args()

    #Convert the grid if necessary
    if (args.create_grid):
        create_grid(args.grid_xyz, args.grid_bin)
  
    time = args.time

    ist = args.iterations[0]
    ied = args.iterations[1]
    isk = args.iterations[2]
  
    ave = args.average

    fdir = args.directory
    adir = args.average_directory

    #Write the xdmf file
    if (time):
        write_xdmf_time(ist, ied, isk, fdir)
    else:
        for i in xrange(ist, ied, isk):
            write_xdmf(i, fdir)

    if (ave > 0):
        for i in xrange(ist, ied, isk):
            write_average(i, ave, adir)
            write_average_xdmf(i, ave)

    return

def write_xdmf_time(tis, tie, ts, fdir):

    #Need to get sizes
    with h5py.File('grid.h5', 'r') as f:
         imax, jmax, kmax = f['x'].shape
  
    print imax, jmax, kmax
    xfile = 'xtime%06d_0.xdmf' % (tis)
    gfile = '%s' % ('grid.h5')

    with open(xfile, 'w') as f:
        f.write("<?xml version=\"1.0\" ?>\n")
        f.write("<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n")
        f.write("<Xdmf Version=\"2.0\">\n")

        f.write(" <Domain>\n")
        f.write("  <Grid Name=\"Time\" GridType=\"Collection\" CollectionType=\"Temporal\">\n")
        for i in xrange(tis, tie, ts):
            ufile = os.path.join(fdir, '%s%06d_0.h5' % ('ufield', i))



            f.write("   <Grid Name=\"grid\" GridType=\"Uniform\">\n")
            f.write("   <Time Value=\"%f\"/>\n" % (float(i)))
            f.write("     <Topology TopologyType=\"3DSMesh\" NumberOfElements=\"%d %d %d\"/>\n" % (imax, jmax, kmax))
            f.write("     <Geometry GeometryType=\"X_Y_Z\">\n")
            f.write("       <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
                    " Precision=\"8\" Format=\"HDF\">\n" %  (imax, jmax, kmax))
            f.write("        %s:/x\n" % (gfile));
            f.write("       </DataItem>\n");
            f.write("       <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
                    " Precision=\"8\" Format=\"HDF\">\n" %  (imax, jmax, kmax))
            f.write("        %s:/y\n" %(gfile))
            f.write("       </DataItem>\n")
            f.write("       <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
                    " Precision=\"8\" Format=\"HDF\">\n" %  (imax, jmax, kmax))
            f.write("        %s:/z\n" %(gfile))
            f.write("       </DataItem>\n")
            f.write("     </Geometry>\n")
            f.write("     <Attribute Name=\"Velocity\" AttributeType=\"Vector\" Center=\"Node\">\n");
            f.write("       <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d 3\" "
                    " Type=\"HyperSlab\">\n" % (imax,jmax,kmax))
            f.write("         <DataItem Dimensions=\"3 4\" Format=\"XML\">\n")
            f.write("             0 0 0 0\n")
            f.write("             1 1 1 1\n")
            f.write("             %d %d %d 3\n" %(imax, jmax, kmax))
            f.write("         </DataItem>\n")
            f.write("         <DataItem Dimensions=\"%d %d %d 3\" NumberType=\"Float\" "
                    " Precision=\"8\" Format=\"HDF\">\n" % (imax+1, jmax+1, kmax+1))
            f.write("           %s:/ucat\n" % (ufile));
            f.write("         </DataItem>\n");
            f.write("       </DataItem>\n");
            f.write("     </Attribute>\n");
            f.write("   </Grid>\n");

        f.write("  </Grid>\n");
        f.write(" </Domain>\n");
        f.write("</Xdmf>\n");


def write_xdmf(ti, fdir):
    
    #Need to get sizes
    with h5py.File('grid.h5', 'r') as f:
         imax, jmax, kmax = f['x'].shape
  
    print imax, jmax, kmax
    xfile = 'xfield%06d_0.xdmf' % (ti)
    gfile = '%s' % ('grid.h5')
    ufile = os.path.join(fdir, '%s%06d_0.h5' % ('ufield', ti))
    vfile = os.path.join(fdir,'%s%06d_0.h5' % ('vfield', ti))
    pfile = os.path.join(fdir,'%s%06d_0.h5' % ('pfield', ti))
    nvfile = os.path.join(fdir,'%s%06d_0.h5' % ('nvfield', ti))

    with open(xfile, 'w') as f:
       f.write("<?xml version=\"1.0\" ?>\n")
       f.write("<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n")
       f.write("<Xdmf Version=\"2.0\">\n")
       f.write(" <Domain>\n")
       f.write("   <Grid Name=\"grid\" GridType=\"Uniform\">\n")
       f.write("     <Topology TopologyType=\"3DSMesh\" NumberOfElements=\"%d %d %d\"/>\n" % \
               (imax, jmax, kmax))
       f.write("     <Geometry GeometryType=\"X_Y_Z\">\n")
       f.write("       <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
               " Precision=\"8\" Format=\"HDF\">\n" %  (imax, jmax, kmax))
       f.write("        %s:/x\n" % (gfile));
       f.write("       </DataItem>\n");
       f.write("       <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
               " Precision=\"8\" Format=\"HDF\">\n" %  (imax, jmax, kmax))
       f.write("        %s:/y\n" %(gfile))
       f.write("       </DataItem>\n")
       f.write("       <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
               " Precision=\"8\" Format=\"HDF\">\n" %  (imax, jmax, kmax))
       f.write("        %s:/z\n" %(gfile))
       f.write("       </DataItem>\n")
       f.write("     </Geometry>\n")
       f.write("     <Attribute Name=\"Pressure\" AttributeType=\"Scalar\" Center=\"Node\">\n");
       f.write("       <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d\" "
                         " Type=\"HyperSlab\">\n" % (imax,jmax,kmax))
       f.write("         <DataItem Dimensions=\"3 3\" Format=\"XML\">\n")
       f.write("             0 0 0 \n")
       f.write("             1 1 1 \n")
       f.write("             %d %d %d \n" %(imax, jmax, kmax))
       f.write("         </DataItem>\n")
       f.write("         <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
                           " Precision=\"8\" Format=\"HDF\">\n" % (imax+1,jmax+1,kmax+1))
       f.write("           %s:/pressure\n" %(pfile))
       f.write("         </DataItem>\n")
       f.write("       </DataItem>\n")
       f.write("     </Attribute>\n")
       f.write("     <Attribute Name=\"Velocity\" AttributeType=\"Vector\" Center=\"Node\">\n");
       f.write("       <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d 3\" "
                         " Type=\"HyperSlab\">\n" % (imax,jmax,kmax))
       f.write("         <DataItem Dimensions=\"3 4\" Format=\"XML\">\n")
       f.write("             0 0 0 0\n")
       f.write("             1 1 1 1\n")
       f.write("             %d %d %d 3\n" %(imax, jmax, kmax))
       f.write("         </DataItem>\n")
       f.write("         <DataItem Dimensions=\"%d %d %d 3\" NumberType=\"Float\" "
                           " Precision=\"8\" Format=\"HDF\">\n" % (imax+1, jmax+1, kmax+1))
       f.write("           %s:/ucat\n" % (ufile));
       f.write("         </DataItem>\n");
       f.write("       </DataItem>\n");
       f.write("     </Attribute>\n");
       f.write("     <Attribute Name=\"Contra-Vel\" AttributeType=\"Vector\" Center=\"Node\">\n");
       f.write("       <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d 3\" "
                         " Type=\"HyperSlab\">\n" % (imax,jmax,kmax))
       f.write("         <DataItem Dimensions=\"3 4\" Format=\"XML\">\n")
       f.write("             0 0 0 0\n")
       f.write("             1 1 1 1\n")
       f.write("             %d %d %d 3\n" %(imax, jmax, kmax))
       f.write("         </DataItem>\n")
       f.write("         <DataItem Dimensions=\"%d %d %d 3\" NumberType=\"Float\" "
                           " Precision=\"8\" Format=\"HDF\">\n" % (imax+1, jmax+1, kmax+1))
       f.write("          %s:/ucont\n" % (vfile));
       f.write("         </DataItem>\n");
       f.write("       </DataItem>\n");
       f.write("     </Attribute>\n");
       f.write("     <Attribute Name=\"Nvert\" AttributeType=\"Scalar\" Center=\"Node\">\n");
       f.write("       <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d\" "
                         " Type=\"HyperSlab\">\n" % (imax,jmax,kmax))
       f.write("         <DataItem Dimensions=\"3 3\" Format=\"XML\">\n")
       f.write("             0 0 0\n")
       f.write("             1 1 1\n")
       f.write("             %d %d %d\n" %(imax, jmax, kmax))
       f.write("         </DataItem>\n")
       f.write("         <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
                           " Precision=\"8\" Format=\"HDF\">\n" % (imax+1, jmax+1, kmax+1))
       f.write("           %s:/nvert\n" % (nvfile));
       f.write("         </DataItem>\n");
       f.write("       </DataItem>\n");
       f.write("     </Attribute>\n");
       f.write("   </Grid>\n");
       f.write(" </Domain>\n");
       f.write("</Xdmf>\n");

    

def create_grid(isxyz, isbin):

    #Read the grid
    if (isxyz):
        fn = 'xyz.dat'
    else:
        fn = 'grid.dat'

    print 'Reading: ', fn 
    f = open(fn, 'r') 

    if (isxyz):
        pass
    elif (isbin):
        f.read(4)
    else:
        f.readline()

    #read the array sizes
    if (isbin):
        imax = struct.unpack('i', f.read(4))[0]
        jmax = struct.unpack('i', f.read(4))[0]
        kmax = struct.unpack('i', f.read(4))[0]
    else:
        (imax, jmax, kmax) = f.readline().split() 
        imax = np.int(imax) 
        jmax = np.int(jmax)
        kmax = np.int(kmax)

    print 'Size: ', imax, jmax, kmax
    #Create Grid structure
    gr = np.zeros((kmax, jmax, imax, 3))

    #Read Binary, easy just read from file
    if (isbin):
        gr[:,:,:,0] = np.fromfile(f, count=(imax*jmax*kmax), 
                                 dtype=np.float64).reshape((kmax,jmax,imax))
        gr[:,:,:,1] = np.fromfile(f, count=(imax*jmax*kmax), 
                                 dtype=np.float64).reshape((kmax,jmax,imax))   
        gr[:,:,:,2] = np.fromfile(f, count=(imax*jmax*kmax), 
                                  dtype=np.float64).reshape((kmax,jmax,imax))    
    #Read xyz, it is cartesian formed by three vectors
    elif (isxyz):
        x = np.zeros(imax)
        y = np.zeros(jmax)
        z = np.zeros(kmax)

        for i in xrange(imax):
            x[i], _, _ = f.readline().split()
        for j in xrange(jmax):
            _, y[j], _ = f.readline().split()
        for k in xrange(kmax):
            _, _, z[k] = f.readline().split()

        for k in xrange(kmax):
            for j in xrange(jmax):
                for i in xrange(imax):
                    gr[k,j,i,0] = x[i]
        for k in xrange(kmax):
            for j in xrange(jmax):
                for i in xrange(imax):
                    gr[k,j,i,1] = y[j]
        for k in xrange(kmax):
            for j in xrange(jmax):
                for i in xrange(imax):
                    gr[k,j,i,2] = z[k]
    #Read a ascii, need to go through each point
    else:
        gr[:,:,:,0] = np.fromfile(f, count=(imax*jmax*kmax), sep=" ",
                                 dtype=np.float64).reshape((kmax,jmax,imax))
        gr[:,:,:,1] = np.fromfile(f, count=(imax*jmax*kmax), sep=" ",
                                 dtype=np.float64).reshape((kmax,jmax,imax))   
        gr[:,:,:,2] = np.fromfile(f, count=(imax*jmax*kmax), sep=" ",
                                  dtype=np.float64).reshape((kmax,jmax,imax))    
 
    f.close()
    

    print 'Finish Reading'

    #Create the h5 grid      
    hfn = 'grid.h5'
    print 'Writing: ',hfn

    with h5py.File(hfn, "w") as f:
         f.create_dataset("x", data=gr[:,:,:,0])
         f.create_dataset("y", data=gr[:,:,:,1])
         f.create_dataset("z", data=gr[:,:,:,2])
         

    return    

def write_average(ti, ave, adir):

    afn = 'a0_field%06d_0.h5' % (ti)
    afn2 = 'a1_field%06d_0.h5' % (ti)
    afn3 = 'a2_field%06d_0.h5' % (ti)

    with h5py.File(os.path.join(adir,'su0_%06d_0.h5'% (ti)), 'r') as su0:
        u = su0['usum'][...]
    with h5py.File(os.path.join(adir,'su1_%06d_0.h5'%(ti)), 'r') as su1:
        uv = su1['ucross'][...]
    with h5py.File(os.path.join(adir,'su2_%06d_0.h5'%(ti)), 'r') as su2:
        uu = su2['usquare'][...]
    with h5py.File(os.path.join(adir,'sp_%06d_0.h5'%(ti)), 'r') as sp:
        p = sp[sp.keys()[0]][...]

    u /= np.float64(ti)
    uu = uu/np.float64(ti) - u**2
    
    uv[:,:,:,0] = uv[:,:,:,0]/np.float64(ti) - u[:,:,:,0]*u[:,:,:,1]
    uv[:,:,:,1] = uv[:,:,:,1]/np.float64(ti) - u[:,:,:,1]*u[:,:,:,2]
    uv[:,:,:,2] = uv[:,:,:,2]/np.float64(ti) - u[:,:,:,2]*u[:,:,:,0]

    p /= np.float64(ti)

    with h5py.File(afn, 'w') as f:
        f.create_dataset('usum', data=u)
        f.create_dataset('ucross', data=uv)
        f.create_dataset('usquare', data=uu)
        f.create_dataset('psum', data=p)


    if (ave < 2):
        return

    with h5py.File(os.path.join(adir,'sp2_%06d_0.h5'%(ti)), 'r') as sp2:
        pp = sp2[sp2.keys()[0]][...]
   
    pp = pp/np.float64(ti) - p**2
   
    with h5py.File(afn2, 'w') as f:
        f.create_dataset('psquare', data=pp)
   
    if (ave < 3):
        return

    with h5py.File(os.path.join(adir,'su3_%06d_0.h5'%(ti)), 'r') as su3:
        udp = su3['udpsum'][...]
    with h5py.File(os.path.join(adir,'su4_%06d_0.h5'%(ti)), 'r') as su4:
        du2 = su4['du2sum'][...]
    with h5py.File(os.path.join(adir,'su5_%06d_0.h5'%(ti)), 'r') as su5:
        uuu = su5['uuusum'][...]
    with h5py.File(os.path.join(adir,'svo_%06d_0.h5'%(ti)), 'r') as sv:
        v = sv['vortsum'][...]
    with h5py.File(os.path.join(adir,'svo2_%06d_0.h5'%(ti)), 'r') as sv2:
        vv = sv2['vortsquare'][...]
    
        
    udp /= np.float64(ti)
    du2 /= np.float64(ti)
    uuu /= np.float64(ti)
    v /= np.float64(ti)
    vv = vv/np.float64(ti) - v**2

    with h5py.File(afn3, 'w') as f:
        f.create_dataset('udpsum', data=udp)
        f.create_dataset('du2sum', data=du2)
        f.create_dataset('uuusum', data=uuu)
        f.create_dataset('vortsum', data=v)
        f.create_dataset('vortsquare', data=vv)

    return


def write_average_xdmf(ti, ave):
    
    #Need to get sizes
    with h5py.File('grid.h5', 'r') as f:
         imax, jmax, kmax = f['x'].shape
  
    print imax, jmax, kmax
    xfile = 'xavefield%06d_0.xdmf' % (ti)
    gfile = '%s' % ('grid.h5')
    a0file = '%s%06d_0.h5' % ('a0_field', ti)
    a1file = '%s%06d_0.h5' % ('a1_field', ti)
    a2file = '%s%06d_0.h5' % ('a2_field', ti)

    with open(xfile, 'w') as f:
       f.write("<?xml version=\"1.0\" ?>\n")
       f.write("<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n")
       f.write("<Xdmf Version=\"2.0\">\n")
       f.write(" <Domain>\n")
       f.write("   <Grid Name=\"grid\" GridType=\"Uniform\">\n")
       f.write("     <Topology TopologyType=\"3DSMesh\" NumberOfElements=\"%d %d %d\"/>\n" % \
               (imax, jmax, kmax))
       f.write("     <Geometry GeometryType=\"X_Y_Z\">\n")
       f.write("       <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
               " Precision=\"8\" Format=\"HDF\">\n" %  (imax, jmax, kmax))
       f.write("        %s:/x\n" % (gfile));
       f.write("       </DataItem>\n");
       f.write("       <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
               " Precision=\"8\" Format=\"HDF\">\n" %  (imax, jmax, kmax))
       f.write("        %s:/y\n" %(gfile))
       f.write("       </DataItem>\n")
       f.write("       <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
               " Precision=\"8\" Format=\"HDF\">\n" %  (imax, jmax, kmax))
       f.write("        %s:/z\n" %(gfile))
       f.write("       </DataItem>\n")
       f.write("     </Geometry>\n")
       f.write("     <Attribute Name=\"Ave Velocity\" AttributeType=\"Vector\" Center=\"Node\">\n");
       f.write("       <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d 3\" "
                         " Type=\"HyperSlab\">\n" % (imax,jmax,kmax))
       f.write("         <DataItem Dimensions=\"3 4\" Format=\"XML\">\n")
       f.write("             0 0 0 0\n")
       f.write("             1 1 1 1\n")
       f.write("             %d %d %d 3\n" %(imax, jmax, kmax))
       f.write("         </DataItem>\n")
       f.write("         <DataItem Dimensions=\"%d %d %d 3\" NumberType=\"Float\" "
                           " Precision=\"8\" Format=\"HDF\">\n" % (imax+1, jmax+1, kmax+1))
       f.write("           %s:/usum\n" % (a0file));
       f.write("         </DataItem>\n");
       f.write("       </DataItem>\n");
       f.write("     </Attribute>\n");
       f.write("     <Attribute Name=\"Cross Velocity\" AttributeType=\"Vector\" Center=\"Node\">\n");
       f.write("       <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d 3\" "
                         " Type=\"HyperSlab\">\n" % (imax,jmax,kmax))
       f.write("         <DataItem Dimensions=\"3 4\" Format=\"XML\">\n")
       f.write("             0 0 0 0\n")
       f.write("             1 1 1 1\n")
       f.write("             %d %d %d 3\n" %(imax, jmax, kmax))
       f.write("         </DataItem>\n")
       f.write("         <DataItem Dimensions=\"%d %d %d 3\" NumberType=\"Float\" "
                           " Precision=\"8\" Format=\"HDF\">\n" % (imax+1, jmax+1, kmax+1))
       f.write("          %s:/ucross\n" % (a0file));
       f.write("         </DataItem>\n");
       f.write("       </DataItem>\n");
       f.write("     </Attribute>\n");
       f.write("     <Attribute Name=\"Square Velocity\" AttributeType=\"Vector\" Center=\"Node\">\n");
       f.write("       <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d 3\" "
                         " Type=\"HyperSlab\">\n" % (imax,jmax,kmax))
       f.write("         <DataItem Dimensions=\"3 4\" Format=\"XML\">\n")
       f.write("             0 0 0 0\n")
       f.write("             1 1 1 1\n")
       f.write("             %d %d %d 3\n" %(imax, jmax, kmax))
       f.write("         </DataItem>\n")
       f.write("         <DataItem Dimensions=\"%d %d %d 3\" NumberType=\"Float\" "
                           " Precision=\"8\" Format=\"HDF\">\n" % (imax+1, jmax+1, kmax+1))
       f.write("          %s:/usquare\n" % (a0file));
       f.write("         </DataItem>\n");
       f.write("       </DataItem>\n");
       f.write("     </Attribute>\n");
       f.write("     <Attribute Name=\"Ave Pressure\" AttributeType=\"Scalar\" Center=\"Node\">\n");
       f.write("       <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d\" "
                         " Type=\"HyperSlab\">\n" % (imax,jmax,kmax))
       f.write("         <DataItem Dimensions=\"3 3\" Format=\"XML\">\n")
       f.write("             0 0 0\n")
       f.write("             1 1 1\n")
       f.write("             %d %d %d\n" %(imax, jmax, kmax))
       f.write("         </DataItem>\n")
       f.write("         <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
                           " Precision=\"8\" Format=\"HDF\">\n" % (imax+1, jmax+1, kmax+1))
       f.write("           %s:/psum\n" % (a0file));
       f.write("         </DataItem>\n");
       f.write("       </DataItem>\n");
       f.write("     </Attribute>\n");

       if (ave < 2):
           f.write("   </Grid>\n");
           f.write(" </Domain>\n");
           f.write("</Xdmf>\n");
  
           return
    
       f.write("     <Attribute Name=\"Square Pressure\" AttributeType=\"Scalar\" Center=\"Node\">\n");
       f.write("       <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d\" "
                         " Type=\"HyperSlab\">\n" % (imax,jmax,kmax))
       f.write("         <DataItem Dimensions=\"3 3\" Format=\"XML\">\n")
       f.write("             0 0 0\n")
       f.write("             1 1 1\n")
       f.write("             %d %d %d\n" %(imax, jmax, kmax))
       f.write("         </DataItem>\n")
       f.write("         <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
                           " Precision=\"8\" Format=\"HDF\">\n" % (imax+1, jmax+1, kmax+1))
       f.write("           %s:/psquare\n" % (a1file));
       f.write("         </DataItem>\n");
       f.write("       </DataItem>\n");
       f.write("     </Attribute>\n");
   
       if (ave < 3):
           f.write("   </Grid>\n");
           f.write(" </Domain>\n");
           f.write("</Xdmf>\n");
  
           return

       f.write("     <Attribute Name=\"Ave Udp\" AttributeType=\"Scalar\" Center=\"Node\">\n");
       f.write("       <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d\" "
                         " Type=\"HyperSlab\">\n" % (imax,jmax,kmax))
       f.write("         <DataItem Dimensions=\"3 3\" Format=\"XML\">\n")
       f.write("             0 0 0\n")
       f.write("             1 1 1\n")
       f.write("             %d %d %d\n" %(imax, jmax, kmax))
       f.write("         </DataItem>\n")
       f.write("         <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
                           " Precision=\"8\" Format=\"HDF\">\n" % (imax+1, jmax+1, kmax+1))
       f.write("           %s:/udpsum\n" % (a2file));
       f.write("         </DataItem>\n");
       f.write("       </DataItem>\n");
       f.write("     </Attribute>\n");
   
       f.write("     <Attribute Name=\"Ave dU2\" AttributeType=\"Scalar\" Center=\"Node\">\n");
       f.write("       <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d\" "
                         " Type=\"HyperSlab\">\n" % (imax,jmax,kmax))
       f.write("         <DataItem Dimensions=\"3 3\" Format=\"XML\">\n")
       f.write("             0 0 0\n")
       f.write("             1 1 1\n")
       f.write("             %d %d %d\n" %(imax, jmax, kmax))
       f.write("         </DataItem>\n")
       f.write("         <DataItem Dimensions=\"%d %d %d\" NumberType=\"Float\" "
                           " Precision=\"8\" Format=\"HDF\">\n" % (imax+1, jmax+1, kmax+1))
       f.write("           %s:/du2sum\n" % (a2file));
       f.write("         </DataItem>\n");
       f.write("       </DataItem>\n");
       f.write("     </Attribute>\n");
   
       f.write("     <Attribute Name=\"Ave UUU\" AttributeType=\"Vector\" Center=\"Node\">\n");
       f.write("       <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d 3\" "
                         " Type=\"HyperSlab\">\n" % (imax,jmax,kmax))
       f.write("         <DataItem Dimensions=\"3 4\" Format=\"XML\">\n")
       f.write("             0 0 0 0\n")
       f.write("             1 1 1 1\n")
       f.write("             %d %d %d 3\n" %(imax, jmax, kmax))
       f.write("         </DataItem>\n")
       f.write("         <DataItem Dimensions=\"%d %d %d 3\" NumberType=\"Float\" "
                           " Precision=\"8\" Format=\"HDF\">\n" % (imax+1, jmax+1, kmax+1))
       f.write("          %s:/uuusum\n" % (a2file));
       f.write("         </DataItem>\n");
       f.write("       </DataItem>\n");
       f.write("     </Attribute>\n");

       f.write("     <Attribute Name=\"Ave Vorticity\" AttributeType=\"Vector\" Center=\"Node\">\n");
       f.write("       <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d 3\" "
                         " Type=\"HyperSlab\">\n" % (imax,jmax,kmax))
       f.write("         <DataItem Dimensions=\"3 4\" Format=\"XML\">\n")
       f.write("             0 0 0 0\n")
       f.write("             1 1 1 1\n")
       f.write("             %d %d %d 3\n" %(imax, jmax, kmax))
       f.write("         </DataItem>\n")
       f.write("         <DataItem Dimensions=\"%d %d %d 3\" NumberType=\"Float\" "
                           " Precision=\"8\" Format=\"HDF\">\n" % (imax+1, jmax+1, kmax+1))
       f.write("          %s:/vortsum\n" % (a2file));
       f.write("         </DataItem>\n");
       f.write("       </DataItem>\n");
       f.write("     </Attribute>\n");
       f.write("     <Attribute Name=\"Square Vorticity\" AttributeType=\"Vector\" Center=\"Node\">\n");
       f.write("       <DataItem ItemType=\"HyperSlab\" Dimensions=\"%d %d %d 3\" "
                         " Type=\"HyperSlab\">\n" % (imax,jmax,kmax))
       f.write("         <DataItem Dimensions=\"3 4\" Format=\"XML\">\n")
       f.write("             0 0 0 0\n")
       f.write("             1 1 1 1\n")
       f.write("             %d %d %d 3\n" %(imax, jmax, kmax))
       f.write("         </DataItem>\n")
       f.write("         <DataItem Dimensions=\"%d %d %d 3\" NumberType=\"Float\" "
                           " Precision=\"8\" Format=\"HDF\">\n" % (imax+1, jmax+1, kmax+1))
       f.write("          %s:/vortsquare\n" % (a2file));
       f.write("         </DataItem>\n");
       f.write("       </DataItem>\n");
       f.write("     </Attribute>\n");

       f.write("   </Grid>\n");
       f.write(" </Domain>\n");
       f.write("</Xdmf>\n");
  
       return




if __name__ == '__main__':
     main()
