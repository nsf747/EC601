// Generated by gencpp from file realsense2_camera/DeviceInfoRequest.msg
// DO NOT EDIT!


#ifndef REALSENSE2_CAMERA_MESSAGE_DEVICEINFOREQUEST_H
#define REALSENSE2_CAMERA_MESSAGE_DEVICEINFOREQUEST_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace realsense2_camera
{
template <class ContainerAllocator>
struct DeviceInfoRequest_
{
  typedef DeviceInfoRequest_<ContainerAllocator> Type;

  DeviceInfoRequest_()
    {
    }
  DeviceInfoRequest_(const ContainerAllocator& _alloc)
    {
  (void)_alloc;
    }







  typedef boost::shared_ptr< ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator> const> ConstPtr;

}; // struct DeviceInfoRequest_

typedef ::realsense2_camera::DeviceInfoRequest_<std::allocator<void> > DeviceInfoRequest;

typedef boost::shared_ptr< ::realsense2_camera::DeviceInfoRequest > DeviceInfoRequestPtr;
typedef boost::shared_ptr< ::realsense2_camera::DeviceInfoRequest const> DeviceInfoRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


} // namespace realsense2_camera

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d41d8cd98f00b204e9800998ecf8427e";
  }

  static const char* value(const ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd41d8cd98f00b204ULL;
  static const uint64_t static_value2 = 0xe9800998ecf8427eULL;
};

template<class ContainerAllocator>
struct DataType< ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "realsense2_camera/DeviceInfoRequest";
  }

  static const char* value(const ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n"
;
  }

  static const char* value(const ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream&, T)
    {}

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct DeviceInfoRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream&, const std::string&, const ::realsense2_camera::DeviceInfoRequest_<ContainerAllocator>&)
  {}
};

} // namespace message_operations
} // namespace ros

#endif // REALSENSE2_CAMERA_MESSAGE_DEVICEINFOREQUEST_H
